"""
modeling_qwen3_vl.py

Qwen3-VL 模型结构 (DySeg-DPP 压缩集成)。
压缩算法在 compression_unified.py 中实现, 本文件仅包含模型结构和薄封装。

架构特点:
  - 全局注意力 ViT (无 window attention)
  - DeepStack: 多层 ViT 特征注入 LLM
  - q_norm / k_norm (Qwen3 特有)
  - 无 sliding window

变更记录:
  - [方案A] LLM 内部剪枝改为 output_attentions 方式:
    Trend Observe 和 Hard Pruning 不再手动计算 Q/K,
    而是让 decoder layer 正常 forward 返回 attn_weights (含 q_norm/k_norm/RoPE),
    确保 attention score 完全正确。
"""

from typing import Callable, Optional, Union, List, Tuple
import math
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs, is_flash_attn_available
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, is_torchdynamo_compiling
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
    Qwen3VLVisionAttention,
    Qwen3VLVisionBlock,
    Qwen3VLVisionModel,
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
    Qwen3VLTextAttention,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextModel,
    Qwen3VLForConditionalGeneration,
    repeat_kv,
)

from .compression_unified import (
    _td_logger,
    tensor_decomp_compression_qwen,
    query_guided_pruning_qwen3,
    _deepstack_process,
)

def Qwen3VLTextModel_forward(
    self: Qwen3VLTextModel,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    # args for deepstack
    visual_pos_masks: Optional[torch.Tensor] = None,
    deepstack_visual_embeds: Optional[list] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Union[tuple, BaseModelOutputWithPast]:
    """Qwen3-VL TextModel forward, 集成:
    - DeepStack 逐层视觉特征注入
    - 多层 Attention Trend Observe (通过 output_attentions 获取真实 attention score)
    - 单层 Hard Pruning (基于上一层真实 attention score + 趋势辅助, 物理移除 token)
    - Hard Pruning 后同步裁剪 DeepStack embeds 和 visual_pos_masks

    执行顺序 (每一层):
      decoder_layer(output_attentions) → DeepStack inject → trend observe / hard prune
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    # torch.jit.trace() doesn't support cache objects in the output
    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache(config=self.config)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = position_ids[0]

    attention_mask = create_causal_mask(
        config=self.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=text_position_ids,
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # ---- DySeg-DPP pruning config ----
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

    # 解析 query_prune_layer (单层, 整数)
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
            else:
                _td_logger.info(
                    f"[QueryPrune] Single-layer pruning: layer={_qp_layer}, "
                    f"current={_current_visual}, target={target_budget}"
                )

    # 解析 soft_prune_layer，只支持整数:
    #   - 负数 / -1: 禁用
    #   - 正整数 N: 硬剪枝层前 N 层 (如 N=4, _qp_layer=28 → {24,25,26,27})
    _soft_layers = set()
    if td_config is not None and visual_token_range is not None:
        _raw_sp = getattr(td_config, 'soft_prune_layer', -1)
        _sp_int = int(float(_raw_sp)) if isinstance(_raw_sp, (str, int, float)) else -1
        if _sp_int > 0 and _qp_layer > 0:
            # 不含 _qp_layer-1: 该层 attn 已通过 _last_prune_attn 进入 scores,
            # 若再放进 attn_history 会被双重计入 (Fix-Bug2)
            _soft_layers = set(range(max(0, _qp_layer - _sp_int), max(0, _qp_layer - 1)))
        if _soft_layers:
            _td_logger.info(
                f"[TrendObserve] Observing attention trend at layers={sorted(_soft_layers)}, "
                f"hard pruning at layer={_qp_layer}"
            )

    is_prefill = inputs_embeds.shape[1] > 1
    _attn_history = []  # 缓存各观测层的 attention score (B, N_vis)
    _obs_prune_method = getattr(td_config, 'llm_prune_method', 'text_token') if td_config else 'text_token'
    _pruned_done = False  # Fix-Bug4: 防止重入硬剪枝

    # 需要 output_attentions 的层: soft_layers + (qp_layer - 1)
    _need_attn_layers = set(_soft_layers)
    if _qp_layer > 0:
        _need_attn_layers.add(_qp_layer - 1)
    _last_prune_attn = None  # 存储 qp_layer-1 的 attn_scores 用于剪枝

    for layer_idx, decoder_layer in enumerate(self.layers):

        # 决定是否需要 attention weights
        _need_attn = (
            layer_idx in _need_attn_layers
            and is_prefill
            and visual_token_range is not None
        )
        if _need_attn:
            kwargs["output_attentions"] = True
            kwargs["prune_method"] = _obs_prune_method
            kwargs["visual_token_start"] = visual_token_range[0]
            kwargs["visual_token_end"] = visual_token_range[1]
            # 方案A: 默认 True, 用 RoPE 后的 Q/K (含 q_norm + k_norm + RoPE),
            # 与文件头注释一致; 仅当 td_config 显式置 False 时才退化 (Fix-Bug3)
            kwargs["use_rope_for_pruning"] = getattr(td_config, 'use_rope_for_pruning', True) if td_config else True
        else:
            kwargs.pop("output_attentions", None)
            kwargs.pop("prune_method", None)
            kwargs.pop("visual_token_start", None)
            kwargs.pop("visual_token_end", None)
            kwargs.pop("use_rope_for_pruning", None)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = layer_outputs[0]

        # ── DeepStack: 在前 N 层注入 ViT 中间层特征 ──
        if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
            if visual_pos_masks is not None:
                hidden_states = _deepstack_process(
                    hidden_states, visual_pos_masks, deepstack_visual_embeds[layer_idx]
                )

        # ── Attention Trend Observe: 直接使用 decoder layer 返回的 attn_weights ──
        if (
            layer_idx in _soft_layers
            and _need_attn
            and layer_outputs[1] is not None
        ):
            _attn_history.append(layer_outputs[1])  # (B, V) — 经过 q_norm/k_norm/RoPE 的正确值
            _td_logger.info(
                f"[TrendObserve] Layer {layer_idx} ({_obs_prune_method}): score range="
                f"[{layer_outputs[1].min().item():.6f}, {layer_outputs[1].max().item():.6f}]"
            )

        # 保存 qp_layer-1 的 attn_weights 用于剪枝
        if (
            _qp_layer > 0
            and layer_idx == _qp_layer - 1
            and _need_attn
            and layer_outputs[1] is not None
        ):
            _last_prune_attn = layer_outputs[1]  # (B, V)

        # ── Hard Pruning: 使用上一层的真实 attention scores ──
        if (
            layer_idx == _qp_layer
            and is_prefill
            and not _pruned_done
            and visual_token_range is not None
            and hidden_states.shape[1] > visual_token_range[1]
        ):
            _v_start_now, _v_end_now = visual_token_range
            _num_visual_now = _v_end_now - _v_start_now

            if _num_visual_now > target_budget:
                hidden_states, attention_mask, position_ids, text_position_ids, cache_position, num_pruned, _keep_idx = (
                    query_guided_pruning_qwen3(
                        hidden_states=hidden_states,
                        visual_token_range=visual_token_range,
                        target_budget=target_budget,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        text_position_ids=text_position_ids,
                        cache_position=cache_position,
                        past_key_values=past_key_values,
                        td_config=td_config,
                        position_embeddings=position_embeddings,
                        attn_scores=_last_prune_attn,
                        attn_history=_attn_history if _attn_history else None,
                    )
                )

                if num_pruned > 0:
                    # Update visual_token_range
                    visual_token_range = (_v_start_now, _v_start_now + target_budget)
                    if td_config is not None:
                        td_config._visual_token_range = visual_token_range

                    # Rebuild RoPE
                    if position_ids is not None and position_ids.ndim == 3:
                        position_embeddings = self.rotary_emb(hidden_states, position_ids)

                    # Fix DynamicCache._seen_tokens
                    if past_key_values is not None and hasattr(past_key_values, '_seen_tokens'):
                        past_key_values._seen_tokens = hidden_states.shape[1]

                    # Fix-Bug1: 清空 _qp_layer 之后所有层的 KV cache
                    # 否则后续层 update() 时会把新 K/V append 到已存在的旧 cache 末尾,
                    # 导致 cache 长度翻倍 (S_new + S_new = 2 * S_new)
                    if past_key_values is not None and hasattr(past_key_values, 'key_cache'):
                        for _l_after in range(layer_idx + 1, len(past_key_values.key_cache)):
                            if past_key_values.key_cache[_l_after].numel() > 0:
                                past_key_values.key_cache[_l_after] = past_key_values.key_cache[_l_after][:, :, :0, :]
                                past_key_values.value_cache[_l_after] = past_key_values.value_cache[_l_after][:, :, :0, :]

                    # Rebuild causal_mask (Qwen3-VL: 无 sliding window)
                    attention_mask = create_causal_mask(
                        config=self.config,
                        input_embeds=hidden_states,
                        attention_mask=attention_mask,
                        cache_position=cache_position,
                        past_key_values=None,
                        position_ids=text_position_ids,
                    )

                    # ── 同步裁剪 DeepStack embeds + visual_pos_masks ──
                    if _keep_idx is not None:
                        # 裁剪 visual_pos_masks
                        if visual_pos_masks is not None:
                            # 重建: prefix + kept_visual + suffix
                            _vpm_prefix = visual_pos_masks[:, :_v_start_now]
                            _vpm_suffix = visual_pos_masks[:, _v_end_now:]
                            _vpm_visual = visual_pos_masks[:, _v_start_now:_v_end_now]
                            _vpm_kept = _vpm_visual[:, _keep_idx]
                            visual_pos_masks = torch.cat([_vpm_prefix, _vpm_kept, _vpm_suffix], dim=1)

                        # 裁剪 DeepStack embeds: _keep_idx 是 visual-local index。
                        # 为避免 DeepStack list 与 LLM layer_idx 不在同一 index 空间，
                        # 对所有 DeepStack entries 同步裁剪到当前 visual token 集合。
                        if deepstack_visual_embeds is not None:
                            for _ds_i in range(len(deepstack_visual_embeds)):
                                deepstack_visual_embeds[_ds_i] = deepstack_visual_embeds[_ds_i][_keep_idx]

                    # 剪枝后需要更新 _need_attn_layers 中的 visual_token_range
                    # (后续层如果还在 _need_attn_layers 中, 需要用新的 range)
                    # 但由于 hard pruning 只执行一次 (单层), 后续不再需要 attn
                    _pruned_done = True  # Fix-Bug4: 标记已剪枝

    hidden_states = self.norm(hidden_states)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )



def Qwen3VLVisionAttention_forward(
    self: Qwen3VLVisionAttention,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[tuple] = None,
    return_logits: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Qwen3-VL Vision Attention forward。

    当 kwargs 中包含 compute_cls_attn=True 时, 根据 cls_attn_method 计算 CLS attention:
      - "pseudo_cls": 构造伪 CLS = patch 均值, 计算 CLS->patch attention row
      - "col_mean": 注意力矩阵列均值
      - "adaptive" / "col_mean_l2": 同时计算两套

    当 return_logits=True 时, 额外返回逐帧注意力权重 (供 FlashVID 兼容)。
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

    # ---- CLS attention 计算 (DySeg-DPP) ----
    compute_cls_attn = kwargs.pop('compute_cls_attn', False)
    cls_attn_method = kwargs.pop('cls_attn_method', 'pseudo_cls')
    td_config_ref = kwargs.pop('td_config', None)

    if compute_cls_attn and td_config_ref is not None:
        with torch.no_grad():
            head_dim = query_states.shape[-1]
            num_heads = query_states.shape[1]
            cls_scores = torch.zeros(seq_length, device=hidden_states.device, dtype=torch.float32)
            _need_both = (cls_attn_method == 'adaptive' or cls_attn_method == 'col_mean_l2')
            if _need_both:
                cls_scores_pseudo = torch.zeros(seq_length, device=hidden_states.device, dtype=torch.float32)
                cls_scores_colmean = torch.zeros(seq_length, device=hidden_states.device, dtype=torch.float32)

            num_segments = cu_seqlens.shape[0] - 1
            for seg_idx in range(num_segments):
                seg_start = cu_seqlens[seg_idx].item()
                seg_end = cu_seqlens[seg_idx + 1].item()
                seg_len = seg_end - seg_start
                if seg_len <= 0:
                    continue

                Q_seg = query_states[seg_start:seg_end]
                K_seg = key_states[seg_start:seg_end]

                if cls_attn_method == 'pseudo_cls' or _need_both:
                    seg_hidden = hidden_states[seg_start:seg_end]
                    pseudo_cls = seg_hidden.mean(dim=0, keepdim=True)
                    qkv_cls = self.qkv(pseudo_cls)
                    qkv_cls = qkv_cls.reshape(1, 3, num_heads, head_dim).permute(1, 0, 2, 3)
                    q_cls, _, _ = qkv_cls.unbind(0)
                    attn_logits_pseudo = torch.bmm(
                        q_cls.squeeze(0).unsqueeze(1),
                        K_seg.transpose(0, 1).transpose(-2, -1)
                    ) / math.sqrt(head_dim)
                    attn_weights_pseudo = attn_logits_pseudo.softmax(dim=-1)
                    seg_scores_pseudo = attn_weights_pseudo.squeeze(1).mean(dim=0)

                    if cls_attn_method == 'pseudo_cls':
                        cls_scores[seg_start:seg_end] = seg_scores_pseudo
                    if _need_both:
                        cls_scores_pseudo[seg_start:seg_end] = seg_scores_pseudo

                if cls_attn_method == 'col_mean' or _need_both:
                    attn_logits_col = torch.bmm(
                        Q_seg.transpose(0, 1),
                        K_seg.transpose(0, 1).transpose(-2, -1)
                    ) / math.sqrt(head_dim)
                    attn_weights_col = attn_logits_col.softmax(dim=-1)
                    seg_scores_colmean = attn_weights_col.mean(dim=-2).mean(dim=0)

                    if cls_attn_method == 'col_mean':
                        cls_scores[seg_start:seg_end] = seg_scores_colmean
                    if _need_both:
                        cls_scores_colmean[seg_start:seg_end] = seg_scores_colmean

            if _need_both:
                td_config_ref._vit_cls_attn_pseudo = cls_scores_pseudo
                td_config_ref._vit_cls_attn_colmean = cls_scores_colmean
                td_config_ref._vit_cls_attn = cls_scores_pseudo
            else:
                td_config_ref._vit_cls_attn = cls_scores

    # ---- Flash Attention 2 ----
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
        **kwargs,
    )

    # ---- 可选: 返回注意力权重 (用于 FastV 兼容) ----
    attn_weights = None
    if return_logits:
        num_frames = cu_seqlens.shape[0] - 1
        q, k = query_states.squeeze(0), key_states.squeeze(0)
        q, k = q.transpose(0, 1), k.transpose(0, 1)
        q = q.reshape(num_frames, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(num_frames, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / self.head_dim**0.5
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = attn_weights.mean(1).mean(1)

    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    attn_output = self.proj(attn_output)
    return attn_output, attn_weights


def Qwen3VLVisionBlock_forward(
    self: Qwen3VLVisionBlock,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[tuple] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Qwen3-VL VisionBlock forward, 透传 kwargs 到 attention。"""
    residual = hidden_states
    hidden_states, attn_weights = self.attn(
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
    return hidden_states, attn_weights


def Qwen3VLVisionModel_forward(
    self: Qwen3VLVisionModel,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    **kwargs,
) -> Tuple[torch.Tensor, list, Optional[torch.Tensor]]:
    """Qwen3-VL ViT forward。

    与 Qwen2.5-VL 的关键差异:
    1. 纯全局注意力, 无 window attention -> 无 window_index / cu_window_seqlens
    2. 新增 DeepStack: 在指定中间层收集特征, 通过 deepstack_merger_list 投影
    3. CLS attention 在最后一层计算 (所有层都是全局的, 选最后一层即可)

    Returns:
        hidden_states: (post_merger_seq, D) merged 视觉特征
        deepstack_feature_lists: list of (post_merger_seq, D) DeepStack 特征
        attn_weights: Optional 最后一层注意力权重 (用于 FastV 兼容)
    """
    hidden_states = self.patch_embed(hidden_states)

    # Qwen3-VL: fast positional embedding interpolation
    pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
    hidden_states = hidden_states + pos_embeds

    rotary_pos_emb = self.rot_pos_emb(grid_thw)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    # Qwen3-VL: 纯全局注意力, cu_seqlens 就是按帧累加
    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    td_config = getattr(self, 'td_config', None)
    num_blocks = len(self.blocks)
    # CLS attention: 在最后一个 block 计算 (Qwen3-VL 所有层都是全局注意力)
    last_block_idx = num_blocks - 1


    # ---- DeepStack 收集 ----
    deepstack_feature_lists = []

    for layer_num, blk in enumerate(self.blocks):
        return_logits = (layer_num == last_block_idx)

        # CLS attention kwargs
        extra_kwargs = {}
        _cls_method = getattr(td_config, 'cls_attn_method', 'pseudo_cls') if td_config else 'pseudo_cls'
        if layer_num == last_block_idx and td_config is not None and _cls_method != 'l2_norm':
            extra_kwargs['compute_cls_attn'] = True
            extra_kwargs['cls_attn_method'] = _cls_method
            extra_kwargs['td_config'] = td_config

        hidden_states, attn_weights = blk(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            return_logits=return_logits,
            **extra_kwargs,
        )

        # DeepStack: 收集中间层特征
        if hasattr(self, 'deepstack_visual_indexes') and layer_num in self.deepstack_visual_indexes:
            ds_idx = self.deepstack_visual_indexes.index(layer_num)
            deepstack_feature = self.deepstack_merger_list[ds_idx](hidden_states)
            deepstack_feature_lists.append(deepstack_feature)

        # ================================================================

    # ---- 计算 ViT 内部 L2 norm (pre-merger) ----
    if td_config is not None:
        with torch.no_grad():
            td_config._vit_l2_norms = hidden_states.float().norm(dim=-1)

    # ---- 计算 ViT 内部帧级 spread (pre-merger) ----
    if td_config is not None:
        with torch.no_grad():
            _smu_spread = self.spatial_merge_unit
            _n_sg = hidden_states.shape[0] // _smu_spread
            _sg_feats = hidden_states.float().view(_n_sg, _smu_spread, -1).mean(dim=1)
            # Qwen3-VL 无 window 重排: 已是原始顺序, 无需 argsort
            _sg_feats_norm = F.normalize(_sg_feats, dim=-1)

            _cu_sg = (cu_seqlens.float() / _smu_spread).long()
            _n_frames_sp = _cu_sg.shape[0] - 1
            _vit_spread = torch.zeros(_n_frames_sp, device=hidden_states.device, dtype=torch.float32)

            for _fidx in range(_n_frames_sp):
                _f_start = _cu_sg[_fidx].item()
                _f_end = _cu_sg[_fidx + 1].item()
                _f_feat = _sg_feats_norm[_f_start:_f_end]
                if _f_feat.shape[0] > 1:
                    _sim = _f_feat @ _f_feat.T
                    _ut_mask = torch.triu(torch.ones_like(_sim, dtype=torch.bool), diagonal=1)
                    _vit_spread[_fidx] = 1.0 - _sim[_ut_mask].mean()

            td_config._vit_spread = _vit_spread

    # ---- merger (spatial merge) ----
    hidden_states = self.merger(hidden_states)
    # Qwen3-VL 无 window 重排: 不需要 reverse_indices

    # CLS attn: spatial merge unit 分组均值 (无需 reverse)
    if td_config is not None and hasattr(td_config, '_vit_cls_attn') and td_config._vit_cls_attn is not None:
        smu = self.spatial_merge_unit
        cls_attn_raw = td_config._vit_cls_attn
        _cur_seq = cls_attn_raw.shape[0]
        n_groups = _cur_seq // smu
        td_config._vit_cls_attn = cls_attn_raw.view(n_groups, smu).mean(dim=1)

    if td_config is not None and hasattr(td_config, '_vit_cls_attn_pseudo') and td_config._vit_cls_attn_pseudo is not None:
        smu = self.spatial_merge_unit
        _raw_p = td_config._vit_cls_attn_pseudo
        _raw_c = td_config._vit_cls_attn_colmean
        _n_g = _raw_p.shape[0] // smu
        td_config._vit_cls_attn_pseudo = _raw_p.view(_n_g, smu).mean(dim=1)
        td_config._vit_cls_attn_colmean = _raw_c.view(_n_g, smu).mean(dim=1)

    if td_config is not None and hasattr(td_config, '_vit_l2_norms') and td_config._vit_l2_norms is not None:
        smu = self.spatial_merge_unit
        _l2_raw = td_config._vit_l2_norms
        td_config._vit_l2_norms = _l2_raw.view(_l2_raw.shape[0] // smu, smu).mean(dim=1)

    # 记录 fusion split sizes

    return hidden_states, deepstack_feature_lists, attn_weights



# ============================================================================
# Qwen3-VL get_video_features / get_image_features
# ============================================================================


def Qwen3VLModel_get_video_features(
    self: Qwen3VLModel,
    pixel_values_videos: torch.FloatTensor,
    video_grid_thw: Optional[torch.LongTensor] = None,
):
    """get_video_features: 返回 (video_embeds_list, deepstack_embeds, attn_weights)。"""
    pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
    video_embeds, deepstack_video_embeds, attn_weights = self.visual(
        pixel_values_videos, grid_thw=video_grid_thw
    )

    split_sizes = (
        video_grid_thw.prod(-1) // self.visual.spatial_merge_size ** 2
    ).tolist()

    video_embeds = torch.split(video_embeds, split_sizes)
    return video_embeds, deepstack_video_embeds, attn_weights


def Qwen3VLModel_get_image_features(
    self: Qwen3VLModel,
    pixel_values: torch.FloatTensor,
    image_grid_thw: Optional[torch.LongTensor] = None,
):
    """get_image_features: 返回 (image_embeds_list, deepstack_embeds)。"""
    pixel_values = pixel_values.type(self.visual.dtype)
    image_embeds, deepstack_image_embeds, _ = self.visual(
        pixel_values, grid_thw=image_grid_thw
    )
    split_sizes = (
        image_grid_thw.prod(-1) // self.visual.spatial_merge_size ** 2
    ).tolist()
    image_embeds = torch.split(image_embeds, split_sizes)
    return image_embeds, deepstack_image_embeds

# Qwen3-VL LLM Attention Forward (Qwen3 特有: q_norm / k_norm)
# ============================================================================


def Qwen3VLTextAttention_forward(
    self: Qwen3VLTextAttention,
    hidden_states: torch.Tensor,
    position_embeddings: tuple,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple:
    """Qwen3-VL LLM Attention forward (with q_norm/k_norm)。

    新增: 当 kwargs 中 output_attentions=True 且 Flash Attention 2 不返回 attn_weights 时,
    手动计算 pruning 所需的 attention scores。
    此时 query_states/key_states 已经过 q_norm + k_norm + RoPE, 结果完全正确。
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # 保存 RoPE 前的 Q/K (用于 use_rope_for_pruning=False 的消融)
    _q_before_rope = query_states
    _k_before_rope = key_states

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    # 提取 pruning 相关 kwargs (不传给 attention_interface)
    _output_attentions = kwargs.pop("output_attentions", False)
    _prune_method = kwargs.pop("prune_method", "last_token")
    _v_start = kwargs.pop("visual_token_start", 0)
    _v_end = kwargs.pop("visual_token_end", 0)
    _use_rope_for_pruning = kwargs.pop("use_rope_for_pruning", False)

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    # ---- Fallback: 手动计算 pruning attn_weights (Flash Attention 2 不返回) ----
    if _output_attentions and attn_weights is None:
        # 根据 use_rope_for_pruning 选择 Q/K 来源:
        #   True  → 用 RoPE 后的 query_states/key_states (含 q_norm + k_norm + RoPE)
        #   False → 用 RoPE 前的 _q_before_rope/_k_before_rope (仅含 q_norm + k_norm)
        if _use_rope_for_pruning:
            _q_src = query_states
            _k_src = key_states
        else:
            _q_src = _q_before_rope
            _k_src = _k_before_rope

        with torch.no_grad():
            key_states_for_score = repeat_kv(_k_src, self.num_key_value_groups)
            # 只取 visual token 对应的 K 列
            k_visual = key_states_for_score[:, :, _v_start:_v_end, :]  # (B, H, V, d)

            if _prune_method == "last_token":
                q_for_score = _q_src[:, :, -1:, :]  # (B, H, 1, d)
            elif _prune_method == "text_token":
                q_prefix = _q_src[:, :, :_v_start, :]
                q_suffix = _q_src[:, :, _v_end:, :]
                q_for_score = torch.cat([q_prefix, q_suffix], dim=2)  # (B, H, T_text, d)
            else:  # all_token
                q_for_score = _q_src  # (B, H, S, d)

            _scale = self.head_dim ** -0.5
            _logits = torch.matmul(q_for_score, k_visual.transpose(-2, -1)) * _scale  # (B, H, Q, V)
            _weights = _logits.softmax(dim=-1, dtype=torch.float32)

            # 聚合: last_token 直接 squeeze, 其他 max over query -> mean over heads -> (B, V)
            if _prune_method == "last_token":
                attn_weights = _weights[:, :, 0, :].mean(dim=1)  # (B, V)
            else:
                attn_weights = _weights.max(dim=2).values.mean(dim=1)  # (B, V)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def Qwen3VLTextDecoderLayer_forward(
    self: Qwen3VLTextDecoderLayer,
    hidden_states: torch.Tensor,
    position_embeddings: tuple,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> tuple:
    """Qwen3-VL Decoder Layer forward。"""
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, attn_weights = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states, attn_weights


@torch.no_grad()
def Qwen3VLForConditionalGeneration_generate(
    self: Qwen3VLForConditionalGeneration,
    **kwargs,
):
    """透明转发到原始 generate 方法。"""
    return self.generate_ori(**kwargs)


def Qwen3VLModel_forward(
    self: Qwen3VLModel,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Union[tuple, Qwen3VLModelOutputWithPast]:
    """Qwen3-VL Model forward, 集成 DySeg-DPP 视频 token 压缩 + DeepStack。

    流程:
    1. ViT -> 提取视觉特征 + DeepStack 中间层特征
    2. DySeg-DPP 压缩 -> 返回 compressed_embeds + kept_global_indices
    3. 用 kept_global_indices 同步裁剪 DeepStack embeds
    4. 构建 visual_pos_masks
    5. 重建序列 -> LLM forward (含 DeepStack 注入 + pruning)
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    td_config = getattr(self, 'td_config', None)

    # 初始化 visual_pos_masks / deepstack (压缩路径会在内部设置)
    visual_pos_masks = None
    deepstack_visual_embeds = None

    # ---- 图像 embedding (不做压缩, 但收集 DeepStack) ----
    image_mask = None
    deepstack_image_embeds = None
    if pixel_values is not None:
        image_embeds, deepstack_image_embeds = Qwen3VLModel_get_image_features(
            self, pixel_values, image_grid_thw
        )
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    # ---- 视频 embedding + 压缩 ----
    video_mask = None
    deepstack_video_embeds = None
    n_video_tokens = 0
    if pixel_values_videos is not None:
        video_embeds_list, deepstack_video_embeds, cls_attention = Qwen3VLModel_get_video_features(
            self, pixel_values_videos, video_grid_thw
        )

        if td_config is not None:
            _vis_ratio = getattr(td_config, 'stage2_retention_ratio', 1.0)
            _llm_ratio = getattr(td_config, 'llm_prune_ratio', 1.0)

            if _vis_ratio >= 1.0:
                # ===== 原生 video scatter + optional LLM prune bookkeeping =====
                video_embeds_cat = torch.cat(video_embeds_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                n_video_tokens = video_embeds_cat.shape[0]
                _, video_mask = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds_cat
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds_cat)

                video_indices = (input_ids[0] == self.config.video_token_id).nonzero(as_tuple=True)[0]
                if video_indices.numel() > 0:
                    video_start = video_indices[0].item()
                    video_end = video_indices[-1].item() + 1
                    td_config._visual_token_range = (video_start, video_end)
                    td_config._target_budget = max(1, int(video_indices.numel() * _llm_ratio))

                pixel_values_videos = None

            else:
                # ===== DySeg-DPP 压缩路径 (FlashVid-style) =====
                # 核心思路: scatter first → get_rope_index on full sequence →
                # DPP compression → select by global indices → 不重建序列

                # Step 1: 计算 position_ids on FULL original sequence
                if position_ids is None:
                    attention_mask_tensor = (
                        attention_mask if not isinstance(attention_mask, dict) else attention_mask.get("full_attention", attention_mask)
                    )
                    if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                        attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                        if attention_mask_tensor.dtype.is_floating_point:
                            attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                            attention_mask_tensor = (1.0 - attention_mask_tensor).int()

                    orig_position_ids, orig_rope_deltas = self.get_rope_index(
                        input_ids,
                        image_grid_thw,
                        video_grid_thw,
                        attention_mask=attention_mask_tensor if isinstance(attention_mask_tensor, torch.Tensor) else attention_mask,
                    )
                    self.rope_deltas = orig_rope_deltas  # DO NOT MODIFY
                else:
                    orig_position_ids = position_ids
                    orig_rope_deltas = getattr(self, 'rope_deltas', None)

                batch_idx = 0
                _video_token_id = self.config.video_token_id
                video_mask_1d_raw = (input_ids[batch_idx] == _video_token_id)
                video_token_indices = video_mask_1d_raw.nonzero(as_tuple=True)[0]

                # Step 2: Scatter 视频 embedding into inputs_embeds (same as native path)
                video_embeds_cat = torch.cat(video_embeds_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                n_video_tokens = video_embeds_cat.shape[0]
                video_mask_3d = video_mask_1d_raw.unsqueeze(0).unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask_3d, video_embeds_cat)

                # Step 2.5: 构建 visual_pos_masks + deepstack (在压缩前, 完整序列上)
                # 仿照 FlashVid: 先在完整序列上构建, 压缩后用 keep_global_indices 索引
                _vid_mask_1d_full = video_mask_1d_raw.unsqueeze(0)  # (1, seq_len)
                if image_mask is not None:
                    _img_mask_1d_full = image_mask[..., 0]  # (1, seq_len)
                    _visual_pos_masks_full = _img_mask_1d_full | _vid_mask_1d_full
                    _deepstack_visual_embeds_full = []
                    _img_joint = _img_mask_1d_full[_visual_pos_masks_full]
                    _vid_joint = _vid_mask_1d_full[_visual_pos_masks_full]
                    _ds_img = deepstack_image_embeds or []
                    _ds_vid = deepstack_video_embeds or []
                    _n_ds = max(len(_ds_img), len(_ds_vid))
                    for _di in range(_n_ds):
                        _embed_joint = torch.zeros(_visual_pos_masks_full.sum(), _ds_img[0].shape[-1] if _ds_img else _ds_vid[0].shape[-1], device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                        if _di < len(_ds_img):
                            _embed_joint[_img_joint] = _ds_img[_di].to(inputs_embeds.dtype)
                        if _di < len(_ds_vid):
                            _embed_joint[_vid_joint] = _ds_vid[_di].to(inputs_embeds.dtype)
                        _deepstack_visual_embeds_full.append(_embed_joint)
                else:
                    _visual_pos_masks_full = _vid_mask_1d_full
                    _deepstack_visual_embeds_full = deepstack_video_embeds

                # Step 3: Run DPP compression, collect kept GLOBAL indices
                keep_visual_global_indices_list = []
                all_kept_local_indices = []  # 用于 DeepStack 同步 (local to concat video)

                video_offset = 0

                for i, v_embed in enumerate(video_embeds_list):
                    T_i = video_grid_thw[i, 0].item()
                    H_i = video_grid_thw[i, 1].item()
                    W_i = video_grid_thw[i, 2].item()
                    merge_size = self.visual.spatial_merge_size
                    N_i_orig = (H_i * W_i) // (merge_size * merge_size)
                    orig_count = T_i * N_i_orig
                    N_i = N_i_orig

                    v_pos_start = video_offset
                    v_pos_end = video_offset + orig_count
                    video_offset += orig_count

                    this_video_indices = video_token_indices[v_pos_start:v_pos_end]
                    all_pos_ids = orig_position_ids[:, batch_idx, this_video_indices]

                    assert all_pos_ids.shape[-1] == T_i * N_i, (
                        f"Position mismatch: got {all_pos_ids.shape[-1]}, expected {T_i * N_i}"
                    )

                    if all_pos_ids.shape[0] == 4:
                        video_pos_ids = all_pos_ids[1:]
                    else:
                        video_pos_ids = all_pos_ids

                    # DySeg-DPP 压缩
                    comp_embeds, comp_positions, kept_indices = tensor_decomp_compression_qwen(
                        video_embeds=v_embed,
                        position_ids_video=video_pos_ids,
                        num_frames=T_i,
                        tokens_per_frame=N_i,
                        config=td_config,
                    )

                    # Convert kept_indices (local to this video) → global indices in full sequence
                    kept_global = this_video_indices[kept_indices]
                    keep_visual_global_indices_list.append(kept_global)

                    # Write compressed embeddings back to inputs_embeds at original global positions
                    inputs_embeds[0, kept_global] = comp_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

                    # For DeepStack sync: local indices relative to concatenated video embeds
                    all_kept_local_indices.append(kept_indices + v_pos_start)

                assert inputs_embeds.shape[0] == 1, "Compression currently supports batch_size=1"

                # Step 4: Build keep_global_indices (prefix + kept video + suffix)
                seq_len = inputs_embeds.shape[1]
                video_start = video_token_indices[0].item()
                video_end = video_token_indices[-1].item() + 1

                prefix_indices = torch.arange(video_start, device=inputs_embeds.device)
                suffix_indices = torch.arange(video_end, seq_len, device=inputs_embeds.device)
                kept_video_indices = torch.cat(keep_visual_global_indices_list, dim=0)
                keep_global_indices = torch.cat([prefix_indices, kept_video_indices, suffix_indices], dim=0).sort().values

                # Step 5: FlashVid-style index selection
                bsz, _, hidden_size = inputs_embeds.shape
                inputs_embeds = torch.gather(
                    inputs_embeds, dim=1,
                    index=keep_global_indices.view(1, -1, 1).expand(bsz, -1, hidden_size)
                )
                position_ids = orig_position_ids[:, :, keep_global_indices]

                if attention_mask is not None:
                    if isinstance(attention_mask, torch.Tensor):
                        attention_mask = attention_mask[:, keep_global_indices]

                cache_position = torch.arange(seq_len, device=inputs_embeds.device)[keep_global_indices]

                # rope_deltas 保持不变 (DO NOT MODIFY)
                self.rope_deltas = orig_rope_deltas

                # ---- FlashVid-style: 同步裁剪 visual_pos_masks + DeepStack ----
                # visual_pos_masks: 直接用 keep_global_indices 从完整序列 mask 中索引
                visual_pos_masks = _visual_pos_masks_full[:, keep_global_indices]

                # deepstack_visual_embeds: 用 kept_local_all (视频段内 anchor 索引) 裁剪 video 部分
                kept_local_all = torch.cat(all_kept_local_indices)
                if _deepstack_visual_embeds_full is not None:
                    if image_mask is not None:
                        # image+video 场景: 需要从 joint embeds 中按新的 visual_pos_masks 重组
                        # 裁剪 video 部分的 DeepStack, image 部分保持不变
                        _ds_vid_pruned = [ds_embed[kept_local_all] for ds_embed in (deepstack_video_embeds or [])]
                        _ds_img = deepstack_image_embeds or []
                        _n_ds = max(len(_ds_img), len(_ds_vid_pruned))
                        # 在压缩后序列中重建 joint embeds
                        _img_mask_compressed = image_mask[:, keep_global_indices, :][..., 0] if image_mask is not None else None
                        _vid_mask_compressed = torch.zeros(inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device)
                        _vid_mask_compressed[:, video_start:video_start + kept_video_indices.shape[0]] = True
                        deepstack_visual_embeds = []
                        for _di in range(_n_ds):
                            _embed_joint = torch.zeros(visual_pos_masks.sum(), _ds_img[0].shape[-1] if _ds_img else _ds_vid_pruned[0].shape[-1], device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                            _img_joint_c = _img_mask_compressed[visual_pos_masks] if _img_mask_compressed is not None else None
                            _vid_joint_c = _vid_mask_compressed[visual_pos_masks]
                            if _di < len(_ds_img) and _img_joint_c is not None:
                                _embed_joint[_img_joint_c] = _ds_img[_di].to(inputs_embeds.dtype)
                            if _di < len(_ds_vid_pruned):
                                _embed_joint[_vid_joint_c] = _ds_vid_pruned[_di].to(inputs_embeds.dtype)
                            deepstack_visual_embeds.append(_embed_joint)
                    else:
                        # # 纯 video 场景: 直接用 kept_local_all 裁剪
                        # deepstack_visual_embeds = [
                        #     ds_embed[kept_local_all] for ds_embed in _deepstack_visual_embeds_full
                        # ]

                        deepstack_visual_embeds = None
                else:
                    deepstack_visual_embeds = None

                # 记录视觉 token 位置 (在压缩后序列中), 供 LLM 剪枝使用
                n_kept_video = kept_video_indices.shape[0]
                new_v_start = video_start  # prefix 长度不变
                new_v_end = video_start + n_kept_video
                td_config._visual_token_range = (new_v_start, new_v_end)

                _llm_ratio = getattr(td_config, 'llm_prune_ratio', 1.0)
                td_config._target_budget = max(1, int(n_kept_video * _llm_ratio))

                # 如果有图像, image_mask 也需要同步裁剪到新序列长度
                if image_mask is not None:
                    image_mask = image_mask[:, keep_global_indices, :]

                pixel_values_videos = None

        else:
            # ===== 无压缩路径 =====
            video_embeds_cat = torch.cat(video_embeds_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            n_video_tokens = video_embeds_cat.shape[0]
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds_cat
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds_cat)

    # ---- 兜底: 如果视频还没处理 ----
    if pixel_values_videos is not None:
        video_embeds_native_list, deepstack_video_embeds, _ = Qwen3VLModel_get_video_features(
            self, pixel_values_videos, video_grid_thw
        )
        video_embeds_native = torch.cat(video_embeds_native_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        n_video_tokens = video_embeds_native.shape[0]
        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds_native
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds_native)

    # ---- 构建 visual_pos_masks (仅非压缩路径) ----
    # 压缩路径已在 Step 5 中设置了 visual_pos_masks 和 deepstack_visual_embeds
    if visual_pos_masks is None:
        deepstack_visual_embeds = None
        if image_mask is not None or video_mask is not None:
            _img_mask_1d = image_mask[..., 0] if image_mask is not None else None
            _vid_mask_1d = video_mask[..., 0] if video_mask is not None else None

            if _img_mask_1d is not None and _vid_mask_1d is not None:
                visual_pos_masks = _img_mask_1d | _vid_mask_1d
                deepstack_visual_embeds = []
                _img_joint = _img_mask_1d[visual_pos_masks]
                _vid_joint = _vid_mask_1d[visual_pos_masks]
                _ds_img = deepstack_image_embeds or []
                _ds_vid = deepstack_video_embeds or []
                _n_ds = max(len(_ds_img), len(_ds_vid))
                for _di in range(_n_ds):
                    _embed_joint = torch.zeros(visual_pos_masks.sum(), _ds_img[0].shape[-1] if _ds_img else _ds_vid[0].shape[-1], device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                    if _di < len(_ds_img):
                        _embed_joint[_img_joint] = _ds_img[_di].to(inputs_embeds.dtype)
                    if _di < len(_ds_vid):
                        _embed_joint[_vid_joint] = _ds_vid[_di].to(inputs_embeds.dtype)
                    deepstack_visual_embeds.append(_embed_joint)
            elif _img_mask_1d is not None:
                visual_pos_masks = _img_mask_1d
                deepstack_visual_embeds = deepstack_image_embeds
            elif _vid_mask_1d is not None:
                visual_pos_masks = _vid_mask_1d
                deepstack_visual_embeds = deepstack_video_embeds

    # ---- 位置 ID 计算 ----
    if position_ids is None:
        attention_mask_tensor = (
            attention_mask if not isinstance(attention_mask, dict) else attention_mask.get("full_attention", attention_mask)
        )
        if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
            attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
            if attention_mask_tensor.dtype.is_floating_point:
                attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()

        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask=attention_mask_tensor if isinstance(attention_mask_tensor, torch.Tensor) else attention_mask,
            )
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    # ---- LLM forward (含 DeepStack + 剪枝) ----
    outputs = Qwen3VLTextModel_forward(
        self.language_model,
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
        **kwargs,
    )

    return Qwen3VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        rope_deltas=self.rope_deltas,
    )