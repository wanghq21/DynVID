"""
modeling_qwen2_5_vl.py

Slim model-structure file for Qwen2.5-VL with DySeg-DPP compression.
All compression algorithms live in compression_unified.py.
ViT Fusion logic has been entirely removed.
"""

from typing import Callable, Optional, Union, List, Tuple
import math
import os
import torch
import torch.nn as nn
from torch.nn import functional as F

from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs, is_flash_attn_available
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, is_torchdynamo_compiling
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
    Qwen2_5_VLAttention,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLTextModel,
    Qwen2_5_VLModel,
    Qwen2_5_VLVisionAttention,
    Qwen2_5_VLVisionBlock,
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLModelOutputWithPast,
    repeat_kv,
)
try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer
except ImportError:
    Qwen2_5_VLDecoderLayer = None

from .compression_unified import (
    _td_logger,
    tensor_decomp_compression_qwen,
    _dyseg_group_frames,
    _compute_group_budget,
)

# ============================================================================
# TextModel forward (with sliding window + soft/hard pruning, no DeepStack)
# ============================================================================


def Qwen2_5_VLDecoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    """Qwen2.5-VL DecoderLayer forward, 透传 output_attentions 并返回 (h, attn_weights).

    官方最新版 DecoderLayer.forward 不接受 output_attentions 也不返回 attn_weights,
    我们必须 monkey-patch 它来支持 LLM 内部剪枝所需的 attention 收集。
    """
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention - pass output_attentions + cache_position through to self_attn
    attn_outputs = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    if isinstance(attn_outputs, tuple):
        hidden_states = attn_outputs[0]
        attn_weights = attn_outputs[1] if len(attn_outputs) > 1 else None
    else:
        hidden_states = attn_outputs
        attn_weights = None

    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return (hidden_states, attn_weights)


def _fastv_prune_qwen(
    hidden_states: torch.Tensor,
    causal_mask: Optional[torch.Tensor],
    attentions_list: List[torch.Tensor],
    cache_position: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    text_position_ids: Optional[torch.Tensor],
    position_embeddings,
    visual_token_range: Tuple[int, int],
    target_budget: int,
    prune_method: str = "last_token",
):
    """FlashVid-style synchronized pruning.

    Inputs
        hidden_states:       (B, S, D)
        causal_mask:         (B, 1, q, k) or None — full_attention mask
        attentions_list:     list of (B, H, q_eff, k_eff) attention weights
                             collected from observe layers (含 qp_layer-1)
        cache_position:      (S,) or None
        position_ids:        (3, B, S) — M-RoPE 3D positions
        text_position_ids:   (B, S) or None
        position_embeddings: (cos, sin), each (B, S, head_dim) or (S, head_dim)
        visual_token_range:  (v_start, v_end)
        target_budget:       保留视觉 token 数
        prune_method:        "last_token" | "text_token" | "all_token"

    Returns
        (hidden_states, causal_mask, position_ids, text_position_ids,
         cache_position, position_embeddings, keep_full_idx, keep_visual_idx)
    """
    B, S, D = hidden_states.shape
    v_start, v_end = visual_token_range
    num_visual = v_end - v_start

    # ---- 1. 多层 attention 聚合到 (B, V) ----
    # 每层 attn shape: (B, H, q_eff, k_eff)
    # k_eff = S (完整 K), 我们只取 V 段
    per_layer_scores = []
    for attn in attentions_list:
        if attn is None:
            continue
        # 取 visual 段的 K
        attn_v = attn[..., v_start:v_end]  # (B, H, q_eff, V)

        # 按 method 聚合 query 维
        q_eff = attn_v.shape[2]
        if prune_method == "last_token":
            # q_eff 应为 1
            scores_layer = attn_v[:, :, -1, :].mean(dim=1)  # (B, V)
        elif prune_method == "text_token":
            # q_eff 是 text token 数 (prefix + suffix)
            # max over query → 每个 visual token 对所有 text token 的最大关注度
            scores_layer = attn_v.max(dim=2).values.mean(dim=1)  # (B, V)
        else:  # all_token
            scores_layer = attn_v.max(dim=2).values.mean(dim=1)  # (B, V)
        per_layer_scores.append(scores_layer)

    if len(per_layer_scores) == 0:
        # 无可用 attention, 不剪枝
        return (hidden_states, causal_mask, position_ids, text_position_ids,
                cache_position, position_embeddings, None, None)

    # 多层 mean
    scores = torch.stack(per_layer_scores, dim=0).mean(dim=0)  # (B, V)
    _td_logger.info(
        f"[FastV-Prune] method={prune_method}, n_layers={len(per_layer_scores)}, "
        f"scores=[{scores.min().item():.6f}, {scores.max().item():.6f}]"
    )

    # ---- 2. Top-k 选择 ----
    k = min(target_budget, num_visual)
    _, top_idx = scores.topk(k, dim=1)
    top_idx = top_idx.sort(dim=1).values  # (B, k)
    keep_visual_idx = top_idx[0]  # (k,) — V 段内相对索引

    # ---- 3. 构造全局 keep_full_idx ----
    device = hidden_states.device
    keep_full_idx = torch.cat([
        torch.arange(0, v_start, device=device, dtype=torch.long),
        v_start + keep_visual_idx,
        torch.arange(v_end, S, device=device, dtype=torch.long),
    ])  # (S - num_pruned,)

    # ---- 4. 同步切片 ----
    new_hidden = hidden_states[:, keep_full_idx, :].contiguous()

    # position_ids: (3, B, S) → (3, B, S_new)
    new_position_ids = position_ids[..., keep_full_idx].contiguous() if position_ids is not None else None

    # text_position_ids: (B, S)
    new_text_position_ids = text_position_ids[..., keep_full_idx].contiguous() if text_position_ids is not None else None

    # cache_position: (S,)
    new_cache_position = cache_position[keep_full_idx].contiguous() if cache_position is not None else None

    # position_embeddings: (cos, sin)
    if position_embeddings is not None:
        cos, sin = position_embeddings
        if cos.ndim == 3:  # (B, S, d) or (3, S, d) for M-RoPE expanded
            new_cos = cos[..., keep_full_idx, :].contiguous()
            new_sin = sin[..., keep_full_idx, :].contiguous()
        elif cos.ndim == 4:  # (3, B, S, d)
            new_cos = cos[..., keep_full_idx, :].contiguous()
            new_sin = sin[..., keep_full_idx, :].contiguous()
        elif cos.ndim == 2:  # (S, d)
            new_cos = cos[keep_full_idx, :].contiguous()
            new_sin = sin[keep_full_idx, :].contiguous()
        else:
            new_cos = cos
            new_sin = sin
        new_position_embeddings = (new_cos, new_sin)
    else:
        new_position_embeddings = None

    # causal_mask: (B, 1, q, k) — 切 q 和 k 两个维度
    if causal_mask is not None:
        if causal_mask.ndim == 4:
            new_causal_mask = causal_mask[:, :, keep_full_idx, :][:, :, :, keep_full_idx].contiguous()
        else:
            new_causal_mask = causal_mask
    else:
        new_causal_mask = None

    return (new_hidden, new_causal_mask, new_position_ids, new_text_position_ids,
            new_cache_position, new_position_embeddings, keep_full_idx, keep_visual_idx)


def Qwen2_5_VLTextModel_forward(
    self: Qwen2_5_VLTextModel,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Union[tuple, BaseModelOutputWithPast]:
    """Qwen2.5-VL TextModel forward, FlashVid-style LLM 内部剪枝。

    在 [qp - soft_n, qp - 1] 多层启用 output_attentions 收集 attn,
    在 layer_idx == qp_layer 进入 layer 之前触发 fastv_prune 同步切片。
    """
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You must specify exactly one of input_ids or inputs_embeds"
        )

    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache(config=self.config)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(
            3, inputs_embeds.shape[0], -1
        )
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(
            3, position_ids.shape[0], -1
        )

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = None

    if not isinstance(causal_mask_mapping := attention_mask, dict):
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": text_position_ids,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        if self.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = (
                create_sliding_window_causal_mask(**mask_kwargs)
            )

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    # ---- FlashVid-style: assert all full_attention, 单一 mask ----
    # 兼容新旧版 transformers: 新版字段名是 layer_type, 旧版是 attention_type
    def _get_layer_attn_type(layer):
        # 优先读 self_attn.layer_type (新版), 退回到 attention_type (旧版/自定义)
        sa = getattr(layer, 'self_attn', None)
        if sa is not None:
            lt = getattr(sa, 'layer_type', None)
            if lt is not None:
                return lt
        return getattr(layer, 'attention_type', 'full_attention')

    assert all(
        _get_layer_attn_type(decoder_layer) == "full_attention"
        for decoder_layer in self.layers[: self.config.num_hidden_layers]
    ), (
        "FlashVid-style LLM pruning requires all decoder layers to be "
        "full_attention. Set use_sliding_window=False in model config."
    )
    causal_mask = causal_mask_mapping["full_attention"]

    # ---- 解析剪枝配置 ----
    td_config = getattr(self, 'td_config', None)
    if td_config is None:
        for parent_attr in ('model', 'language_model'):
            parent = getattr(self, parent_attr, None)
            if parent is not None:
                td_config = getattr(parent, 'td_config', None)
                if td_config is not None:
                    break

    visual_token_range = getattr(td_config, '_visual_token_range', None) if td_config else None
    target_budget = getattr(td_config, '_target_budget', 0) if td_config else 0

    _qp_layer = -1
    if td_config is not None and visual_token_range is not None and target_budget > 0:
        _raw_qp = getattr(td_config, 'query_prune_layer', -1)
        if isinstance(_raw_qp, str):
            _raw_qp = _raw_qp.strip()
            _qp_layer = int(_raw_qp) if _raw_qp else -1
        elif isinstance(_raw_qp, (int, float)):
            _qp_layer = int(_raw_qp)

        if _qp_layer >= 0:
            _v_start, _v_end = visual_token_range
            _current_visual = _v_end - _v_start
            if _current_visual <= target_budget:
                _qp_layer = -1

    # soft_prune_layer: 多层 observe (含 qp_layer-1)
    _soft_n = 0
    if td_config is not None and _qp_layer > 0:
        _raw_sp = getattr(td_config, 'soft_prune_layer', -1)
        _sp_int = int(float(_raw_sp)) if isinstance(_raw_sp, (str, int, float)) else -1
        if _sp_int > 0:
            _soft_n = _sp_int

    # observe_layers: [qp - soft_n, qp - 1] (qp-1 必含)
    _observe_layers = set()
    if _qp_layer > 0:
        _observe_layers.add(_qp_layer - 1)
        if _soft_n > 1:
            _observe_layers.update(range(max(0, _qp_layer - _soft_n), _qp_layer))

    _prune_method = getattr(td_config, 'llm_prune_method', 'text_token') if td_config else 'text_token'

    is_prefill = inputs_embeds.shape[1] > 1

    # ---- 解析剪枝配置 (日志只在 prefill 阶段打印一次) ----
    if is_prefill and _qp_layer > 0:
        _v_start_log, _v_end_log = visual_token_range
        _td_logger.info(
            f"[FastV-Prune] qp_layer={_qp_layer}, "
            f"current={_v_end_log - _v_start_log}, target={target_budget}"
        )
        if _observe_layers:
            _td_logger.info(
                f"[FastV-Prune] observe layers={sorted(_observe_layers)}, "
                f"hard prune at layer={_qp_layer}, soft_n={_soft_n}"
            )

    _attn_history = []  # list of (B, H, q_eff, S)
    _output_attentions = output_attentions

    for layer_idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # ---- 在剪枝层之前触发 fastv_prune ----
        if (
            is_prefill
            and _qp_layer > 0
            and layer_idx == _qp_layer
            and visual_token_range is not None
            and len(_attn_history) > 0
        ):
            _v_start_now, _v_end_now = visual_token_range
            _num_visual_now = _v_end_now - _v_start_now
            if _num_visual_now > target_budget and hidden_states.shape[1] > _v_end_now:
                (
                    hidden_states,
                    causal_mask,
                    position_ids,
                    text_position_ids,
                    cache_position,
                    position_embeddings,
                    keep_full_idx,
                    keep_visual_idx,
                ) = _fastv_prune_qwen(
                    hidden_states=hidden_states,
                    causal_mask=causal_mask,
                    attentions_list=_attn_history,
                    cache_position=cache_position,
                    position_ids=position_ids,
                    text_position_ids=text_position_ids,
                    position_embeddings=position_embeddings,
                    visual_token_range=visual_token_range,
                    target_budget=target_budget,
                    prune_method=_prune_method,
                )

                if keep_visual_idx is not None:
                    new_v_count = keep_visual_idx.shape[0]
                    visual_token_range = (_v_start_now, _v_start_now + new_v_count)
                    if td_config is not None:
                        td_config._visual_token_range = visual_token_range
                    _td_logger.info(
                        f"[FastV-Prune] {_num_visual_now} → {new_v_count} visual tokens "
                        f"(layer {_qp_layer})"
                    )

        # ---- 决定本层是否启用 output_attentions (用于收集 attn) ----
        _need_attn = (
            is_prefill
            and layer_idx in _observe_layers
            and visual_token_range is not None
        )
        if _need_attn:
            kwargs["prune_method"] = _prune_method
            kwargs["visual_token_start"] = visual_token_range[0]
            kwargs["visual_token_end"] = visual_token_range[1]
            kwargs["use_rope_for_pruning"] = getattr(td_config, 'use_rope_for_pruning', False) if td_config else False
            _layer_output_attentions = True
        else:
            kwargs.pop("prune_method", None)
            kwargs.pop("visual_token_start", None)
            kwargs.pop("visual_token_end", None)
            kwargs.pop("use_rope_for_pruning", None)
            _layer_output_attentions = _output_attentions

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            output_attentions=_layer_output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = layer_outputs[0]
        attn_weights = layer_outputs[1] if len(layer_outputs) > 1 else None

        # ---- 收集 attn 用于剪枝 ----
        if _need_attn and attn_weights is not None:
            _attn_history.append(attn_weights)

        if _output_attentions:
            all_self_attns += (attn_weights,)

    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    if not return_dict:
        return tuple(
            v
            for v in [
                hidden_states,
                past_key_values,
                all_hidden_states,
                all_self_attns,
            ]
            if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )



# ============================================================================
# VisionTransformerPretrainedModel forward (ViT Fusion DELETED)
# ============================================================================


def Qwen2_5_VisionTransformerPretrainedModel_forward(
    self: Qwen2_5_VisionTransformerPretrainedModel,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
) -> torch.Tensor:
    """Qwen2.5-VL Vision Transformer forward。

    与原生 forward 完全一致, 仅在最后一个全注意力层额外计算 column-wise mean
    attention (CLS attention 替代) 并存入 td_config._vit_cls_attn。
    """
    hidden_states = self.patch_embed(hidden_states)
    rotary_pos_emb = self.rot_pos_emb(grid_thw)

    window_index, cu_window_seqlens = self.get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
        device=hidden_states.device,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(
        seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
    )
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(
        seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
    )
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

    # 将 window_index 转为 GPU tensor, 供后续 reverse 使用
    if not isinstance(window_index, torch.Tensor):
        window_index = torch.tensor(window_index, device=hidden_states.device, dtype=torch.long)
    else:
        window_index = window_index.to(device=hidden_states.device)

    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    # 找到最后一个全注意力层的索引, 用于计算 col_mean CLS attention
    td_config = getattr(self, 'td_config', None)
    last_fullatt_idx = max(self.fullatt_block_indexes) if self.fullatt_block_indexes else -1

    for layer_num, blk in enumerate(self.blocks):
        if layer_num in self.fullatt_block_indexes:
            cu_seqlens_now = cu_seqlens
        else:
            cu_seqlens_now = cu_window_seqlens

        # 仅在最后一个全注意力层传入 td_config, 触发 col_mean 统计
        extra_kwargs = {}
        if layer_num == last_fullatt_idx and td_config is not None:
            extra_kwargs['td_config'] = td_config

        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens_now,
            position_embeddings=position_embeddings,
            **extra_kwargs,
        )

    hidden_states = self.merger(hidden_states)

    # Reverse window_index 恢复原始顺序
    reverse_indices = torch.argsort(window_index)
    hidden_states = hidden_states[reverse_indices, :]

    # 将 _vit_cls_attn 也按 reverse_indices 恢复原始顺序
    # (col_mean 在 pre-merger 维度算出, 这里做 spatial_merge_unit 分组均值 + reverse)
    if td_config is not None and getattr(td_config, '_vit_cls_attn', None) is not None:
        cls_attn_raw = td_config._vit_cls_attn
        smu = self.spatial_merge_unit
        n_groups = cls_attn_raw.shape[0] // smu
        cls_attn_merged = cls_attn_raw.view(n_groups, smu).mean(dim=1)
        td_config._vit_cls_attn = cls_attn_merged[reverse_indices]

    return hidden_states



# ============================================================================
# VisionBlock forward
# ============================================================================


def Qwen2_5_VLVisionBlock_forward(
    self: Qwen2_5_VLVisionBlock,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> torch.Tensor:
    """Qwen2.5-VL VisionBlock forward。"""
    residual = hidden_states
    hidden_states = self.attn(
        self.norm1(hidden_states),
        cu_seqlens=cu_seqlens,
        rotary_pos_emb=rotary_pos_emb,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.mlp(self.norm2(hidden_states))
    hidden_states = residual + hidden_states
    return hidden_states


# ============================================================================
# VisionAttention forward (CLS attention computation)
# ============================================================================


def Qwen2_5_VLVisionAttention_forward(
    self: Qwen2_5_VLVisionAttention,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> torch.Tensor:
    """Qwen2.5-VL Vision Attention forward。

    与原生 attention forward 完全一致, 仅当 kwargs 中传入 td_config 时,
    额外计算 col_mean CLS attention 并存入 td_config._vit_cls_attn。
    实现采用批量化版本: 所有段等长 (batch=1 + 单视频 + 同分辨率帧),
    一次 batched matmul 替代逐段 for 循环, 数学上与原 col_mean 实现完全等价。
    """
    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        self.qkv(hidden_states)
        .reshape(seq_length, 3, self.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(
        query_states, key_states, cos, sin
    )

    # ---- 额外计算 col_mean CLS attention (仅在传入 td_config 的层) ----
    td_config_ref = kwargs.pop('td_config', None)
    cls_attn_method = kwargs.pop('cls_attn_method', 'col_mean')

    if td_config_ref is not None:
        if cls_attn_method != 'col_mean':
            raise NotImplementedError(
                f"Only cls_attn_method='col_mean' is supported, got '{cls_attn_method}'"
            )
        with torch.no_grad():
            head_dim = query_states.shape[-1]
            num_heads = query_states.shape[1]
            total_tokens = query_states.shape[0]
            T = cu_seqlens.shape[0] - 1
            assert total_tokens % T == 0, \
                f"col_mean 要求等长段: total={total_tokens}, T={T}"
            L = total_tokens // T

            # (T*L, H, d) → (T, H, L, d)
            Q = query_states.view(T, L, num_heads, head_dim).permute(0, 2, 1, 3)
            K = key_states.view(T, L, num_heads, head_dim).permute(0, 2, 1, 3)

            chunk_t = 8
            scores = torch.empty(T, L, device=query_states.device, dtype=torch.float32)
            for s in range(0, T, chunk_t):
                e = min(s + chunk_t, T)
                logits = torch.matmul(Q[s:e], K[s:e].transpose(-2, -1)) / math.sqrt(head_dim)
                w = logits.softmax(dim=-1)
                # 列均值: 先 mean over query 维 (dim=-2), 再 mean over heads (dim=1)
                scores[s:e] = w.mean(dim=-2).mean(dim=1)

            td_config_ref._vit_cls_attn = scores.reshape(T * L).to(torch.float32)
            _td_logger.debug(
                f"[ViT col_mean] T={T}, L={L}, "
                f"range=[{td_config_ref._vit_cls_attn.min().item():.6f}, "
                f"{td_config_ref._vit_cls_attn.max().item():.6f}]"
            )

    # ---- 正常的 Flash Attention forward (与原生一致) ----
    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[
            self.config._attn_implementation
        ]

    assert self.config._attn_implementation == "flash_attention_2"
    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
    attn_output, _ = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask=None,
        scaling=self.scaling,
        dropout=0.0 if not self.training else self.attention_dropout,
        cu_seq_lens_q=cu_seqlens,
        cu_seq_lens_k=cu_seqlens,
        max_length_q=max_seqlen,
        max_length_k=max_seqlen,
        is_causal=False,
    )

    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    attn_output = self.proj(attn_output)
    return attn_output



# ============================================================================
# LLM Attention forward (unmodified)
# ============================================================================


def Qwen2_5_VLAttention_forward(
    self: Qwen2_5_VLAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    """Qwen2.5-VL LLM Attention forward, 支持 output_attentions fallback 用于 pruning。"""
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    # 保存 RoPE 前的 Q/K (用于 use_rope_for_pruning=False 的消融)
    _q_before_rope = query_states
    _k_before_rope = key_states

    cos, sin = position_embeddings
    # 兼容新旧版: 新版 rope_parameters["mrope_section"], 旧版 rope_scaling["mrope_section"]
    if hasattr(self, 'rope_scaling') and self.rope_scaling is not None and "mrope_section" in self.rope_scaling:
        _mrope_section = self.rope_scaling["mrope_section"]
    else:
        _mrope_section = self.config.rope_parameters["mrope_section"]
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        _mrope_section,
    )

    if past_key_values is not None:
        cache_kwargs = {
            "sin": sin,
            "cos": cos,
            "cache_position": cache_position,
        }
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    # 提取 pruning 相关 kwargs (不传给 attention_interface)
    # NOTE: output_attentions 是正式函数参数, 不在 kwargs 中, 必须用 函数参数读取
    _output_attentions = output_attentions
    kwargs.pop("output_attentions", None)  # 防御性 pop, 避免重复传给 attention_interface
    _prune_method = kwargs.pop("prune_method", "last_token")
    _v_start = kwargs.pop("visual_token_start", 0)
    _v_end = kwargs.pop("visual_token_end", 0)
    _use_rope_for_pruning = kwargs.pop("use_rope_for_pruning", False)

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[
            self.config._attn_implementation
        ]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        position_ids=position_ids,
        **kwargs,
    )

    # ---- Fallback: 手动计算 pruning attn_weights (Flash Attention 2 不返回) ----
    # 返回完整 (B, H, q_eff, S) 矩阵, V-segment 切片由 _fastv_prune_qwen 负责
    if _output_attentions and attn_weights is None:
        # 根据 use_rope_for_pruning 选择 Q/K 来源
        if _use_rope_for_pruning:
            _q_src = query_states
            _k_src = key_states
        else:
            _q_src = _q_before_rope
            _k_src = _k_before_rope

        with torch.no_grad():
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import repeat_kv
            key_states_for_score = repeat_kv(_k_src, self.num_key_value_groups)
            # K 保持 full sequence: (B, H, S, d)
            k_full = key_states_for_score

            if _prune_method == "last_token":
                q_for_score = _q_src[:, :, -1:, :]  # (B, H, 1, d)
            elif _prune_method == "text_token":
                q_prefix = _q_src[:, :, :_v_start, :]
                q_suffix = _q_src[:, :, _v_end:, :]
                q_for_score = torch.cat([q_prefix, q_suffix], dim=2)  # (B, H, T_text, d)
            else:  # all_token
                q_for_score = _q_src  # (B, H, S, d)

            _scale = self.head_dim ** -0.5
            _logits = torch.matmul(q_for_score, k_full.transpose(-2, -1)) * _scale
            # 返回 (B, H, q_eff, S) 完整矩阵
            attn_weights = _logits.softmax(dim=-1, dtype=torch.float32)

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights


# ============================================================================
# generate (transparent passthrough)
# ============================================================================


@torch.no_grad()
def Qwen2_5_VLForConditionalGeneration_generate(
    self: Qwen2_5_VLForConditionalGeneration,
    **kwargs,
):
    """透明转发到原始 generate 方法。"""
    return self.generate_ori(**kwargs)


# ============================================================================
# get_video_features (simplified, no ViT Fusion split)
# ============================================================================


def Qwen2_5_VLModel_get_video_features(
    self: Qwen2_5_VLModel,
    pixel_values_videos: torch.FloatTensor,
    video_grid_thw: Optional[torch.LongTensor] = None,
):
    """get_video_features: 标准 split, 无 ViT Fusion。"""
    pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
    video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)

    split_sizes = (
        video_grid_thw.prod(-1) // self.visual.spatial_merge_size ** 2
    ).tolist()

    video_embeds = torch.split(video_embeds, split_sizes)
    return video_embeds



# ============================================================================
# Model forward (ViT Fusion DELETED, simplified compression path)
# ============================================================================


def Qwen2_5_VLModel_forward(
    self: Qwen2_5_VLModel,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Union[tuple, Qwen2_5_VLModelOutputWithPast]:
    """Qwen2.5-VL Model forward, 集成 DySeg + DPP + Softmax Fusion 压缩。

    ViT Fusion 已完全移除。压缩位置: 视频 embedding 提取之后, LLM forward 之前。
    当 vision-side ratio >= 1.0 且 LLM ratio >= 1.0 时, 自动 bypass 压缩路径走原生逻辑。
    """
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    # ---- 图像 embedding (不做压缩) ----
    if pixel_values is not None:
        image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    # ---- 视频 embedding + 压缩 ----
    if pixel_values_videos is not None:
        video_embeds = self.get_video_features(
            pixel_values_videos, video_grid_thw
        )

        td_config = getattr(self, 'td_config', None)

        # === 判断压缩路径 ===
        # Important debug/fix path:
        # If stage2_retention_ratio >= 1.0, do NOT enter the pre-LLM reconstruction
        # path. We keep the native Qwen2.5-VL video scatter and only record the
        # visual token range for optional inner-LLM pruning. This avoids changing
        # position_ids / cache_position / rope_deltas when vision-side compression
        # is disabled.
        _vis_ratio = 1.0
        _llm_ratio = 1.0
        if td_config is not None:
            _vis_ratio = getattr(td_config, 'stage2_retention_ratio', 1.0)
            _llm_ratio = getattr(td_config, 'llm_prune_ratio', 1.0)

        if td_config is not None and _vis_ratio >= 1.0:
            # ===== 原生 video scatter + optional LLM prune bookkeeping =====
            video_embeds_cat = torch.cat(video_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                video_features=video_embeds_cat,
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                video_mask, video_embeds_cat
            )

            # Only record the contiguous video-token range for inner-LLM pruning.
            # Do not manually set position_ids / cache_position / rope_deltas here;
            # let the native get_rope_index branch below compute them.
            video_indices = (input_ids[0] == self.config.video_token_id).nonzero(as_tuple=True)[0]
            if video_indices.numel() > 0:
                video_start = video_indices[0].item()
                video_end = video_indices[-1].item() + 1
                td_config._visual_token_range = (video_start, video_end)
                td_config._target_budget = max(1, int(video_indices.numel() * _llm_ratio))
                _td_logger.info(
                    f"[NativeScatter+LLMPrune] visual=[{video_start},{video_end}), "
                    f"current={video_indices.numel()}, target={td_config._target_budget}, "
                    f"llm_ratio={_llm_ratio}"
                )

            # Prevent the fallback block below from scattering video a second time.
            pixel_values_videos = None

        elif td_config is not None:
            # ===== 压缩路径 (FlashVid-style: 先 scatter 再按 global index 选取) =====
            #
            # 关键: 借鉴 FlashVid 做法, 先将 video embedding scatter 到完整序列,
            # 然后在完整序列上用 get_rope_index 计算 position_ids,
            # 最后按 keep_global_indices 从完整序列中选取保留的 token.
            # 这样 position_ids 直接从原始值中索引, 保持了 M-RoPE 的完整性.
            # rope_deltas 也不修改, 与 FlashVid 一致.

            # Step 1: Scatter video embeddings into inputs_embeds (same as native path)
            video_embeds_cat = torch.cat(video_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                video_features=video_embeds_cat,
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds_cat)

            # Step 2: Run DPP compression on each video to get kept_indices
            batch_idx = 0
            _video_token_id = self.config.video_token_id
            video_mask_1d_raw = (input_ids[batch_idx] == _video_token_id)
            video_token_indices = video_mask_1d_raw.nonzero(as_tuple=True)[0]

            # 计算 position_ids (在完整序列上)
            orig_position_ids, orig_rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.rope_deltas = orig_rope_deltas

            # Step 3: For each video segment, run compression and collect kept global indices
            keep_visual_global_indices_list = []
            video_offset = 0

            for i, v_embed in enumerate(video_embeds):
                T_i = video_grid_thw[i, 0].item()
                H_i = video_grid_thw[i, 1].item()
                W_i = video_grid_thw[i, 2].item()
                merge_size = self.visual.spatial_merge_size
                N_i_orig = (H_i * W_i) // (merge_size * merge_size)
                orig_count = T_i * N_i_orig
                N_i = N_i_orig

                # Position IDs for this video segment
                v_pos_start = video_offset
                v_pos_end = video_offset + orig_count
                video_offset += orig_count

                this_video_indices = video_token_indices[v_pos_start:v_pos_end]

                all_pos_ids = orig_position_ids[:, batch_idx, this_video_indices]

                if all_pos_ids.shape[0] == 4:
                    video_pos_ids = all_pos_ids[1:]         # (3, T*N)
                else:
                    video_pos_ids = all_pos_ids             # (3, T*N)

                comp_embeds, comp_positions, kept_indices = tensor_decomp_compression_qwen(
                    video_embeds=v_embed,
                    position_ids_video=video_pos_ids,
                    num_frames=T_i,
                    tokens_per_frame=N_i,
                    config=td_config,
                )

                # kept_indices 是视频段内的局部索引, 转为全局序列索引
                kept_global = this_video_indices[kept_indices]
                keep_visual_global_indices_list.append(kept_global)

                # 将压缩后的 embedding 写回 inputs_embeds 对应位置
                inputs_embeds[0, kept_global] = comp_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )

            # Step 4: 构建 keep_global_indices (prefix + kept_video + suffix)
            seq_len = inputs_embeds.shape[1]
            video_start = video_token_indices[0].item()
            video_end = video_token_indices[-1].item() + 1

            prefix_indices = torch.arange(video_start, device=inputs_embeds.device)
            suffix_indices = torch.arange(video_end, seq_len, device=inputs_embeds.device)
            kept_video_indices = torch.cat(keep_visual_global_indices_list, dim=0)

            keep_global_indices = torch.cat(
                [prefix_indices, kept_video_indices, suffix_indices], dim=0
            ).sort().values

            # Step 5: FlashVid-style 索引选取
            bsz, _, hidden_size = inputs_embeds.shape
            inputs_embeds = torch.gather(
                inputs_embeds,
                dim=1,
                index=keep_global_indices.view(1, -1, 1).expand(bsz, -1, hidden_size),
            )
            position_ids = orig_position_ids[:, :, keep_global_indices]
            if attention_mask is not None:
                attention_mask = attention_mask[:, keep_global_indices]
            cache_position = torch.arange(seq_len, device=inputs_embeds.device)[keep_global_indices]

            # 记录视觉 token 位置, 供 LLM 剪枝使用
            n_kept_video = kept_video_indices.shape[0]
            new_video_start = video_start
            new_video_end = video_start + n_kept_video
            td_config._visual_token_range = (new_video_start, new_video_end)
            _llm_ratio = getattr(td_config, 'llm_prune_ratio', 1.0)
            td_config._target_budget = max(1, int(n_kept_video * _llm_ratio))

            pixel_values_videos = None

        else:
            # ===== 原生路径 (无压缩 或 bypass) =====
            video_embeds_cat = torch.cat(video_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                video_features=video_embeds_cat,
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                video_mask, video_embeds_cat
            )

    # ---- 兜底 ----
    if pixel_values_videos is not None:
        video_embeds_native = self.get_video_features(
            pixel_values_videos, video_grid_thw
        )
        video_embeds_native = torch.cat(video_embeds_native, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        _, video_mask = self.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            video_features=video_embeds_native,
        )
        inputs_embeds = inputs_embeds.masked_scatter(
            video_mask, video_embeds_native
        )

    # ---- 位置 ID 计算 ----
    if position_ids is None:
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (
                past_key_values is None
                or past_key_values.get_seq_length() == 0
            )
        )
        if (
            prefill_compiled_stage or prefill_noncompiled_stage
        ) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(
                seq_length, device=inputs_embeds.device
            )
            position_ids = position_ids.view(1, 1, -1).expand(
                3, batch_size, -1
            )
            if cache_position is not None:
                delta = (cache_position[0] + self.rope_deltas).to(
                    inputs_embeds.device
                )
            else:
                delta = torch.zeros(
                    (batch_size, seq_length), device=inputs_embeds.device
                )
            delta = delta.repeat_interleave(
                batch_size // delta.shape[0], dim=1
            )
            position_ids = position_ids + delta.to(position_ids.device)

    # ---- LLM forward ----
    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )

    output = Qwen2_5_VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )
    return output if return_dict else output.to_tuple()