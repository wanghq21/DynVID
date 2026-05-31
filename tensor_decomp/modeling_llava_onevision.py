"""
modeling_llava_onevision.py

DySeg + DPP Anchor Selection + Top-K Soft Fusion 视频token压缩 — LLaVA-OneVision / LLaVA-Video 版本。

架构说明:
  LLaVA-OneVision 和 LLaVA-Video 共享同一 `LlavaQwenForCausalLM` 模型类,
  本文件同时适配两者。

Pipeline:
  SigLIP ViT → mm_projector → 2D Pooling (27×27 → 13×13 = 169 tokens/frame)
    → DySeg 分组 → 组预算分配 (Effective Rank / MCD)
    → 组内 DPP anchor 选择 → Top-K 软分配 + 自适应垃圾桶融合
    → Frame-mode newline 插入 → Rebuild embedding list
    → Qwen2 LLM (标准 1D RoPE)
    → (可选) LLM 内部 attention-based 硬剪枝

与 Qwen2.5-VL 版的关键差异:
  - 无 M-RoPE (标准 1D RoPE, position_ids 为 arange)
  - 无 DeepStack
  - 无 ViT 内部 token fusion
  - Token 填入 LLM 用 "rebuild embedding list" 而非 masked_scatter
  - 压缩函数返回 keep_visual_indices (而非 M-RoPE positions)
  - image_newline 在 frame mode 下每帧末尾追加

Monkey-patch targets (7 个):
  1. SigLipAttention.forward           → 返回 per-token importance
  2. SigLipVisionTower.forward         → 返回 (features, cls_attentions)
  3. LlavaMetaForCausalLM.encode_images → 调用修改后的 ViT
  4. LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal → 插入 DySeg-DPP 压缩
  5. Qwen2Attention.forward            → pruning layer 返回 attn_weights
  6. Qwen2DecoderLayer.forward         → 传递 attn_weights
  7. Qwen2Model.forward                → LLM 侧剪枝触发
"""

from __future__ import annotations

import logging
import math
import random
import re
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .compression_unified import (
    tensor_decomp_compression_llava,
    query_guided_pruning_llava,
)

# ============================================================================
# Logger
# ============================================================================
_td_logger = logging.getLogger("tensor_decomp.llava")


# ============================================================================
# ██  Monkey-patch 函数: SigLIP ViT
# ============================================================================

@torch.no_grad()
def SigLipVisionTower_forward(self, images: torch.Tensor):
    """修改 SigLipVisionTower.forward: 返回 (features, cls_attentions)。"""
    if not isinstance(images, torch.Tensor):
        raise ValueError(f"Unexpected data type: {type(images)}")
    image_forward_outs = self.vision_tower(
        images.to(device=self.device, dtype=self.dtype),
        output_attentions=True,
        output_hidden_states=True,
    )
    image_features = image_forward_outs.hidden_states[-1].to(images.dtype)
    cls_attentions = image_forward_outs.attentions[-1].to(images.dtype)
    assert image_features.shape[-2] == 729, (
        f"Expected 729 patches, got {image_features.shape[-2]}"
    )
    return image_features, cls_attentions


def SigLipAttention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """修改 SigLipAttention.forward: 返回 attn_weights.mean(heads).mean(queries)。"""
    batch_size, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    k_v_seq_len = key_states.shape[-2]
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)
    attn_output = self.out_proj(attn_output)

    # 返回 per-token importance: mean over heads, mean over queries
    return attn_output, attn_weights.mean(1).mean(1)


# ============================================================================
# ██  Monkey-patch 函数: LlavaMetaForCausalLM
# ============================================================================

def LlavaMetaForCausalLM_encode_images(self, images: torch.Tensor):
    """调用修改后的 ViT, 返回 (features, cls_attentions)。"""
    image_features, cls_attentions = self.get_model().get_vision_tower()(images)
    image_features = self.get_model().mm_projector(image_features)
    return image_features, cls_attentions


def LlavaMetaForCausalLM_prepare_inputs_labels_for_multimodal(
    self,
    input_ids,
    position_ids,
    attention_mask,
    past_key_values,
    labels,
    images,
    modalities=["image"],
    image_sizes=None,
):
    """核心: 插入 DySeg-DPP 压缩, 替代 FlashVID。

    修改点:
      1. 视频路径: 使用 tensor_decomp_compression_llava 替代 flashvid_compression
      2. Frame-mode newline: 使用 keep_visual_indices 按帧分配
      3. 记录 visual_token_start_index 和 visual_token_length 供 LLM 剪枝使用
    """
    from llava.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX
    from llava.mm_utils import get_anyres_image_grid_shape
    from llava.model.llava_arch import unpad_image
    from llava.utils import rank0_print

    vision_tower = self.get_vision_tower()
    if vision_tower is None or images is None or input_ids.shape[1] == 1:
        return input_ids, position_ids, attention_mask, past_key_values, None, labels

    if isinstance(modalities, str):
        modalities = [modalities]

    td_config = getattr(self, 'td_config', None)

    if type(images) is list or images.ndim == 5:
        if type(images) is list:
            images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

        video_idx_in_batch = []
        for _ in range(len(modalities)):
            if modalities[_] == "video":
                video_idx_in_batch.append(_)

        images_list = []
        for image in images:
            if image.ndim == 4:
                images_list.append(image)
            else:
                images_list.append(image.unsqueeze(0))

        concat_images = torch.cat([image for image in images_list], dim=0)
        split_sizes = [image.shape[0] for image in images_list]
        encoded_image_features, cls_attentions = self.encode_images(concat_images)

        encoded_image_features = torch.split(encoded_image_features, split_sizes)
        image_features = []
        assert len(encoded_image_features) == 1, "Only support single video in a batch for now."

        # 暂存压缩结果供 frame-mode newline 使用
        _compression_results = {}

        for idx, image_feat in enumerate(encoded_image_features):
            if idx in video_idx_in_batch and td_config is not None:
                # ---- DySeg-DPP 压缩 ----
                visual_token_start_index = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].item()

                pooled_image_feature = self.get_2dPool(image_feat)
                pooled_cls_attentions = self.get_2dPool(
                    cls_attentions[:split_sizes[idx]].unsqueeze(-1)
                ).squeeze(-1)

                num_frames, num_visual_tokens = pooled_image_feature.shape[:2]

                # 提取 text embeddings 供 query bonus 使用
                text_token_mask = (input_ids[0] != IMAGE_TOKEN_INDEX)
                if text_token_mask.any():
                    text_ids = input_ids[0][text_token_mask]
                    text_embeds = self.get_model().embed_tokens(text_ids)
                    td_config._text_embeds = text_embeds
                else:
                    td_config._text_embeds = None

                compressed_visual_tokens, keep_visual_indices = tensor_decomp_compression_llava(
                    video_embeds=pooled_image_feature,
                    cls_attention=pooled_cls_attentions,
                    num_frames=num_frames,
                    tokens_per_frame=num_visual_tokens,
                    config=td_config,
                )

                _compression_results[idx] = {
                    'compressed_visual_tokens': compressed_visual_tokens,
                    'keep_visual_indices': keep_visual_indices,
                    'pooled_image_feature': pooled_image_feature,
                    'num_frames': num_frames,
                    'num_visual_tokens': num_visual_tokens,
                    'visual_token_start_index': visual_token_start_index,
                }

                image_features.append(compressed_visual_tokens)
            elif idx in video_idx_in_batch:
                # 无 td_config: 不压缩, 直接 2D pooling
                pooled_image_feature = self.get_2dPool(image_feat)
                image_features.append(pooled_image_feature)
            else:
                image_features.append(image_feat)

        mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
        image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
        mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")

        if mm_patch_merge_type == "flat":
            image_features = [x.flatten(0, 1) for x in image_features]

        elif mm_patch_merge_type.startswith("spatial"):
            new_image_features = []
            for image_idx, image_feature in enumerate(image_features):
                if image_idx in video_idx_in_batch:
                    if mm_newline_position == "grid":
                        if image_idx in _compression_results:
                            # Grid mode 不兼容压缩后的 1D token, fallback 到 frame-mode newline
                            cr = _compression_results[image_idx]
                            compressed_visual_tokens = cr['compressed_visual_tokens']
                            keep_visual_indices = cr['keep_visual_indices']
                            num_frames = cr['num_frames']
                            num_visual_tokens = cr['num_visual_tokens']

                            compressed_visual_token_list = []
                            for frame_idx in range(num_frames):
                                start_idx = frame_idx * num_visual_tokens
                                end_idx = start_idx + num_visual_tokens
                                ind = torch.where(
                                    (keep_visual_indices >= start_idx) & (keep_visual_indices < end_idx)
                                )[0]
                                frame_visual_tokens = compressed_visual_tokens[ind]
                                frame_visual_tokens = torch.cat(
                                    (frame_visual_tokens,
                                     self.model.image_newline[None].to(image_feature.device)),
                                    dim=0,
                                )
                                compressed_visual_token_list.append(frame_visual_tokens)

                            image_feature = torch.cat(compressed_visual_token_list, dim=0)
                            if td_config is not None:
                                td_config._visual_token_range = (
                                    cr['visual_token_start_index'],
                                    cr['visual_token_start_index'] + image_feature.shape[0],
                                )
                                td_config._visual_token_length = image_feature.shape[0]
                        else:
                            image_feature = self.add_token_per_grid(image_feature)
                        new_image_features.append(image_feature)

                    elif mm_newline_position == "frame":
                        # ---- Frame-mode newline: 每帧末尾追加 image_newline ----
                        if image_idx in _compression_results:
                            cr = _compression_results[image_idx]
                            compressed_visual_tokens = cr['compressed_visual_tokens']
                            keep_visual_indices = cr['keep_visual_indices']
                            num_frames = cr['num_frames']
                            num_visual_tokens = cr['num_visual_tokens']

                            compressed_visual_token_list = []
                            for frame_idx in range(num_frames):
                                start_idx = frame_idx * num_visual_tokens
                                end_idx = start_idx + num_visual_tokens
                                ind = torch.where(
                                    (keep_visual_indices >= start_idx) & (keep_visual_indices < end_idx)
                                )[0]
                                frame_visual_tokens = compressed_visual_tokens[ind]
                                frame_visual_tokens = torch.cat(
                                    (frame_visual_tokens,
                                     self.model.image_newline[None].to(image_feature.device)),
                                    dim=0,
                                )
                                compressed_visual_token_list.append(frame_visual_tokens)

                            image_feature = torch.cat(compressed_visual_token_list, dim=0)

                            # 更新 td_config 中的 visual token 信息供 LLM 剪枝使用
                            if td_config is not None:
                                td_config._visual_token_range = (
                                    cr['visual_token_start_index'],
                                    cr['visual_token_start_index'] + image_feature.shape[0],
                                )
                                td_config._visual_token_length = image_feature.shape[0]
                        else:
                            # 未压缩: 原始 frame-mode (flatten)
                            image_feature = image_feature.flatten(0, 1)

                        new_image_features.append(image_feature)

                    elif mm_newline_position == "one_token":
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat(
                                (image_feature, self.model.image_newline[None].to(image_feature.device)),
                                dim=0,
                            )
                        # 压缩后更新 visual token 信息供 LLM 剪枝使用
                        if image_idx in _compression_results and td_config is not None:
                            cr = _compression_results[image_idx]
                            td_config._visual_token_range = (
                                cr['visual_token_start_index'],
                                cr['visual_token_start_index'] + image_feature.shape[0],
                            )
                            td_config._visual_token_length = image_feature.shape[0]
                        new_image_features.append(image_feature)

                    elif mm_newline_position == "no_token":
                        if image_idx in _compression_results:
                            # 压缩后已是 1D, 不需要 flatten
                            if td_config is not None:
                                cr = _compression_results[image_idx]
                                td_config._visual_token_range = (
                                    cr['visual_token_start_index'],
                                    cr['visual_token_start_index'] + image_feature.shape[0],
                                )
                                td_config._visual_token_length = image_feature.shape[0]
                        else:
                            image_feature = image_feature.flatten(0, 1)
                        new_image_features.append(image_feature)
                    else:
                        raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")

                elif image_feature.shape[0] > 1:
                    # Multi-image operations (保持原始逻辑)
                    base_image_feature = image_feature[0]
                    image_feature = image_feature[1:]
                    height = width = self.get_vision_tower().num_patches_per_side
                    assert height * width == base_image_feature.shape[0]

                    if "anyres_max" in image_aspect_ratio:
                        matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                        if matched_anyres_max_num_patches:
                            max_num_patches = int(matched_anyres_max_num_patches.group(1))

                    if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                        if hasattr(self.get_vision_tower(), "image_size"):
                            vision_tower_image_size = self.get_vision_tower().image_size
                        else:
                            raise ValueError("vision_tower_image_size not found")
                        try:
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                                image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size
                            )
                        except Exception as e:
                            rank0_print(f"Error: {e}")
                            num_patch_width, num_patch_height = 2, 2
                        image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                    else:
                        image_feature = image_feature.view(2, 2, height, width, -1)

                    if "maxpool2x2" in mm_patch_merge_type:
                        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = nn.functional.max_pool2d(image_feature, 2)
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                        unit = image_feature.shape[2]
                        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = unpad_image(image_feature, image_sizes[image_idx])
                        c, h, w = image_feature.shape
                        times = math.sqrt(h * w / (max_num_patches * unit**2))
                        if times > 1.1:
                            image_feature = image_feature[None]
                            image_feature = nn.functional.interpolate(
                                image_feature, [int(h // times), int(w // times)], mode="bilinear"
                            )[0]
                        image_feature = torch.cat(
                            (image_feature,
                             self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)),
                            dim=-1,
                        )
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    elif "unpad" in mm_patch_merge_type:
                        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = unpad_image(image_feature, image_sizes[image_idx])
                        image_feature = torch.cat(
                            (image_feature,
                             self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)),
                            dim=-1,
                        )
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    else:
                        image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                        image_feature = image_feature.flatten(0, 3)

                    if "nobase" not in mm_patch_merge_type:
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    new_image_features.append(image_feature)
                else:
                    image_feature = image_feature[0]
                    if "unpad" in mm_patch_merge_type:
                        image_feature = torch.cat(
                            (image_feature, self.model.image_newline[None]),
                            dim=0,
                        )
                    new_image_features.append(image_feature)

            image_features = new_image_features
        else:
            raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
    else:
        image_features = self.encode_images(images)

    if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
        raise NotImplementedError

    _labels = labels
    _position_ids = position_ids
    _attention_mask = attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    if position_ids is None:
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    _input_ids = input_ids
    input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
    labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

    new_input_embeds = []
    new_labels = []
    cur_image_idx = 0

    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        if num_images == 0:
            cur_image_features = image_features[cur_image_idx]
            cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
            cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
            new_input_embeds.append(cur_input_embeds)
            new_labels.append(labels[batch_idx])
            cur_image_idx += 1
            continue

        image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
        cur_input_ids_noim = []
        cur_labels = labels[batch_idx]
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
        split_sizes = [x.shape[0] for x in cur_labels_noim]
        cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
        cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
        cur_new_input_embeds = []
        cur_new_labels = []

        for i in range(num_images + 1):
            cur_new_input_embeds.append(cur_input_embeds_no_im[i])
            cur_new_labels.append(cur_labels_noim[i])
            if i < num_images:
                try:
                    cur_image_features = image_features[cur_image_idx]
                except IndexError:
                    cur_image_features = image_features[cur_image_idx - 1]
                cur_image_idx += 1
                cur_new_input_embeds.append(cur_image_features)
                cur_new_labels.append(
                    torch.full((cur_image_features.shape[0],), IGNORE_INDEX,
                               device=cur_labels.device, dtype=cur_labels.dtype)
                )

        cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
        cur_new_input_embeds = torch.cat(cur_new_input_embeds)
        cur_new_labels = torch.cat(cur_new_labels)

        new_input_embeds.append(cur_new_input_embeds)
        new_labels.append(cur_new_labels)

    tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
    new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
    new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

    max_len = max(x.shape[0] for x in new_input_embeds)
    batch_size = len(new_input_embeds)

    new_input_embeds_padded = []
    new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

    for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
        cur_len = cur_new_embed.shape[0]
        if getattr(self.config, "tokenizer_padding_side", "right") == "left":
            new_input_embeds_padded.append(
                torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]),
                                       dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                           cur_new_embed), dim=0)
            )
            if cur_len > 0:
                new_labels_padded[i, -cur_len:] = cur_new_labels
                attention_mask[i, -cur_len:] = True
                position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
        else:
            new_input_embeds_padded.append(
                torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]),
                                                       dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
            )
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

    new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

    if _labels is None:
        new_labels = None
    else:
        new_labels = new_labels_padded

    if _attention_mask is None:
        attention_mask = None
    else:
        attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

    if _position_ids is None:
        position_ids = None

    if getattr(self.config, "use_pos_skipping", False) and self.training:
        position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0)
        split_position = random.randint(0, new_input_embeds.size(1))
        left_add = random.randint(0, self.config.pos_skipping_range)
        right_add = random.randint(left_add, self.config.pos_skipping_range)
        position_ids[:, :split_position] += left_add
        position_ids[:, split_position:] += right_add

    return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


# ============================================================================
# ██  Monkey-patch 函数: Qwen2 LLM
# ============================================================================

def Qwen2Attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple,
    attention_mask: Optional[torch.Tensor],
    past_key_values=None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> tuple:
    """Qwen2Attention.forward: 在 pruning layer 返回 attn_weights。"""
    from transformers.models.qwen2.modeling_qwen2 import (
        apply_rotary_pos_emb,
        repeat_kv,
    )
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # Use eager or flash attention
    attention_interface = None
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation in ALL_ATTENTION_FUNCTIONS:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    if attention_interface is None:
        from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward
        attention_interface = eager_attention_forward

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=getattr(self, 'sliding_window', None),
        **kwargs,
    )

    # 如果需要 output_attentions 但 FA2 没有返回
    if kwargs.get("output_attentions", False) and attn_weights is None:
        last_query = query_states[:, :, -1:, :]
        key_states_expanded = repeat_kv(key_states, self.num_key_value_groups)
        attn_weights = torch.matmul(last_query, key_states_expanded.transpose(2, 3)) * self.scaling
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def Qwen2DecoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
    """Qwen2DecoderLayer.forward: 返回 attn_weights 给下一层。"""
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


def Qwen2Model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """Qwen2Model.forward: Trend Observe + 硬剪枝触发 (query_guided_pruning_llava)。"""
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    from transformers.modeling_outputs import BaseModelOutputWithPast

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    # Causal mask
    if not isinstance(causal_mask_mapping := attention_mask, dict):
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        if getattr(self, 'has_sliding_layers', False):
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # Get td_config
    td_config = getattr(self, 'td_config', None)
    is_prefill = hidden_states.shape[1] > 1
    _qp_layer = getattr(td_config, 'query_prune_layer', -1) if td_config else -1
    llm_prune_ratio = getattr(td_config, 'llm_prune_ratio', 1.0) if td_config else 1.0

    # 解析 soft_prune_layer，只支持整数:
    #   - 负数 / -1: 禁用
    #   - 正整数 N: 硬剪枝层前 N 层 (如 N=4, _qp_layer=28 → {24,25,26,27})
    _soft_layers = set()
    if td_config is not None:
        _raw_sp = getattr(td_config, 'soft_prune_layer', -1)
        _sp_int = int(float(_raw_sp)) if isinstance(_raw_sp, (str, int, float)) else -1
        if _sp_int > 0 and _qp_layer > 0:
            _soft_layers = set(range(max(0, _qp_layer - _sp_int), _qp_layer))
        if _soft_layers:
            _td_logger.info(
                f"[TrendObserve] Observing attention trend at layers={sorted(_soft_layers)}, "
                f"hard pruning at layer={_qp_layer}"
            )

    # 获取 visual_token_range
    visual_token_range = getattr(td_config, '_visual_token_range', None) if td_config else None

    causal_mask = causal_mask_mapping.get("full_attention", None)

    _attn_history = []  # 缓存各观测层的 attention score (B, N_vis)
    _obs_prune_method = getattr(td_config, 'llm_prune_method', 'text_token') if td_config else 'text_token'

    for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):

        hidden_states, attn_weights = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        # ── Attention Trend Observe: 在指定层观测 attention score ──
        if (
            layer_idx in _soft_layers
            and is_prefill
            and visual_token_range is not None
            and hidden_states.shape[1] > visual_token_range[1]
        ):
            _sv_start, _sv_end = visual_token_range
            _sv_num = _sv_end - _sv_start
            if _sv_num > 0:
                _obs_self_attn = decoder_layer.self_attn
                _obs_head_dim = _obs_self_attn.head_dim
                _obs_num_heads = _obs_self_attn.config.num_attention_heads
                _obs_num_kv_heads = _obs_self_attn.config.num_key_value_heads
                _obs_num_kv_groups = _obs_num_heads // _obs_num_kv_heads

                with torch.no_grad():
                    # 根据 prune_method 确定 query 来源
                    if _obs_prune_method == "last_token":
                        _obs_query_h = hidden_states[:, -1:, :]
                    elif _obs_prune_method == "all_token":
                        _obs_query_h = hidden_states
                    else:
                        # text_token (默认): text prefix + text suffix
                        _obs_text_prefix = hidden_states[:, :_sv_start, :]
                        _obs_text_suffix = hidden_states[:, _sv_end:, :]
                        _obs_query_h = torch.cat([_obs_text_prefix, _obs_text_suffix], dim=1)

                    _obs_q_len = _obs_query_h.shape[1]
                    if _obs_q_len > 0:
                        from transformers.models.qwen2.modeling_qwen2 import repeat_kv as _repeat_kv

                        _obs_visual_h = hidden_states[:, _sv_start:_sv_end, :]

                        _obs_q = _obs_self_attn.q_proj(_obs_query_h)
                        _obs_k = _obs_self_attn.k_proj(_obs_visual_h)

                        _obs_B = hidden_states.shape[0]
                        _obs_q = _obs_q.view(_obs_B, _obs_q_len, _obs_num_heads, _obs_head_dim).transpose(1, 2)
                        _obs_k = _obs_k.view(_obs_B, _sv_num, _obs_num_kv_heads, _obs_head_dim).transpose(1, 2)

                        if _obs_num_kv_groups > 1:
                            _obs_k = _repeat_kv(_obs_k, _obs_num_kv_groups)

                        _obs_scale = 1.0 / math.sqrt(_obs_head_dim)
                        _obs_logits = torch.matmul(_obs_q, _obs_k.transpose(-2, -1)) * _obs_scale
                        _obs_weights = _obs_logits.softmax(dim=-1)

                        # last_token: 只有 1 个 query, 直接取; 其他: max over query tokens
                        if _obs_prune_method == "last_token":
                            _obs_scores = _obs_weights[:, :, 0, :].mean(dim=1)  # (B, N_vis)
                        else:
                            _obs_scores = _obs_weights.max(dim=2).values.mean(dim=1)  # (B, N_vis)

                        _attn_history.append(_obs_scores)
                        _td_logger.info(
                            f"[TrendObserve] Layer {layer_idx} ({_obs_prune_method}): score range="
                            f"[{_obs_scores.min().item():.6f}, {_obs_scores.max().item():.6f}]"
                        )

        # ── Hard Pruning: query-guided 物理移除 token ──
        if (
            layer_idx == _qp_layer
            and is_prefill
            and visual_token_range is not None
            and hidden_states.shape[1] > visual_token_range[1]
        ):
            v_start, v_end = visual_token_range
            num_visual = v_end - v_start

            if llm_prune_ratio < 1.0 and num_visual > 0:
                target_budget = max(1, int(num_visual * llm_prune_ratio))

                (
                    hidden_states,
                    attention_mask_updated,
                    position_ids,
                    cache_position,
                    position_embeddings,
                    num_pruned,
                    keep_idx,
                ) = query_guided_pruning_llava(
                    hidden_states=hidden_states,
                    visual_token_range=visual_token_range,
                    target_budget=target_budget,
                    attention_mask=None,  # 使用 causal_mask
                    position_ids=position_ids,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    td_config=td_config,
                    decoder_layer=decoder_layer,
                    position_embeddings=position_embeddings,
                    attn_history=_attn_history if _attn_history else None,
                )

                if num_pruned > 0:
                    # 更新 visual_token_range
                    new_v_end = v_start + (num_visual - num_pruned)
                    td_config._visual_token_range = (v_start, new_v_end)
                    visual_token_range = (v_start, new_v_end)

                    # 重建 causal_mask
                    new_seq_len = hidden_states.shape[1]
                    if causal_mask is not None:
                        causal_mask = causal_mask[:, :, :new_seq_len, :new_seq_len]

    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
    )