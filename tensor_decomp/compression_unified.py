"""
compression_unified.py

DySeg + DPP Anchor Selection + Top-K Soft Fusion — 统一压缩模块。
供 modeling_qwen3_vl.py, modeling_qwen2_5_vl.py 和 modeling_llava_onevision.py 共同调用。

Pipeline:
  Post-projector 信号计算 → DySeg 分组 → 组预算分配 (Effective Rank / MCD)
  → 组内 DPP anchor 选择 → Top-K 软分配 + 自适应垃圾桶融合
  → (可选) LLM 内部 query-guided 硬剪枝
"""

from typing import Callable, Optional, Union, List, Tuple
import numpy as np
import logging
import math
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import tqdm
from torch import Tensor
from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv


class TqdmHandler(logging.Handler):
    """通过 tqdm.write 输出，避免被进度条覆盖。"""
    def emit(self, record):
        msg = self.format(record)
        tqdm.tqdm.write(msg)

_td_logger = logging.getLogger("tensor_decomp")


def _td_normalize(x: torch.Tensor) -> torch.Tensor:
    """Z-score + sigmoid normalize to ~[0, 1].
    信号区分度低 (σ小) → 全部挤在 0.5 附近 → 乘性融合中不干扰其他信号。
    信号区分度高 (σ大) → 自然拉开差异。
    """
    mu = x.mean()
    sigma = x.std()
    if sigma < 1e-8:
        return torch.full_like(x, 0.5)  # 无区分度 → 中性值
    return torch.sigmoid((x - mu) / sigma)


def _td_normalize_minmax(x: Tensor) -> Tensor:
    """Min-max normalize to [0, 1]."""
    x_min = x.min()
    x_max = x.max()
    if (x_max - x_min) < 1e-8:
        return torch.full_like(x, 0.5)
    return (x - x_min) / (x_max - x_min)


_td_logger.setLevel(logging.INFO)
_td_logger.propagate = False
_local_rank = int(os.environ.get("LOCAL_RANK", 0))
if not _td_logger.handlers and _local_rank == 0:
    _td_tqdm_handler = TqdmHandler()
    _td_tqdm_handler.setFormatter(
        logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    )
    _td_logger.addHandler(_td_tqdm_handler)



# ============================================================================
# 核心算法函数
# ============================================================================

# 原生 DPP 选择函数
def _dpp_select(features: torch.Tensor, importance: torch.Tensor, k: int,
                init_selected: Optional[torch.Tensor] = None) -> torch.Tensor:
    N, D = features.shape
    if k >= N:
        return torch.arange(N, device=features.device)

    feat_norm = F.normalize(features.float(), dim=-1)

    # # min-max 归一化 importance 到 [0, 1] 并 clamp 避免零值
    # imp = importance.float()
    # imp_min, imp_max = imp.min(), imp.max()
    # if (imp_max - imp_min) > 1e-8:
    #     imp_norm = (imp - imp_min) / (imp_max - imp_min)
    # else:
    #     imp_norm = torch.ones_like(imp)
    # imp_norm = imp_norm.clamp(min=1e-4)  # 避免零 importance 导致 kernel 退化
    
    # tdnormalize minmax importance
    imp_norm = _td_normalize_minmax(importance.float())
    imp_norm = imp_norm.clamp(min=1e-4)

    # # tdnormalize importance
    # imp_norm = _td_normalize(importance.float())
    # imp_norm = imp_norm.clamp(min=1e-4)

    B = imp_norm.unsqueeze(-1) * feat_norm  # (N, D)

    # ★ 预计算完整 L 矩阵，一次 GEMM
    L_full = B @ B.T  # (N, N)

    selected = []
    C = torch.zeros(k, N, device=features.device, dtype=torch.float32)
    d = L_full.diagonal().clone()  # L[i,i] = imp[i]^2

    # 处理预选 anchor
    if init_selected is not None and init_selected.numel() > 0:
        n_init = init_selected.shape[0]
        if n_init >= k:
            return init_selected[:k].sort().values
        for j_idx, sel_idx in enumerate(init_selected):
            selected.append(sel_idx)
            L_sel_all = L_full[sel_idx]  # ★ 直接索引，无 matmul
            if j_idx > 0:
                c_prev = C[:j_idx, :]
                c_sel = c_prev[:, sel_idx]
                L_sel_all = L_sel_all - (c_sel @ c_prev)
            sqrt_d = torch.sqrt(d[sel_idx].clamp(min=1e-10))
            C[j_idx] = L_sel_all / sqrt_d
            d = d - C[j_idx] ** 2
            d = d.clamp(min=0)
            d[sel_idx] = 0

    # 贪心选择
    for step in range(len(selected), k):
        best = d.argmax()  # 不 .item()，避免 sync
        selected.append(best)

        L_best_all = L_full[best]  # ★ 直接索引，无 matmul
        if step > 0:
            c_prev = C[:step, :]
            c_best = c_prev[:, best]
            L_best_all = L_best_all - (c_best @ c_prev)
        sqrt_d = torch.sqrt(d[best].clamp(min=1e-10))
        C[step] = L_best_all / sqrt_d
        d = d - C[step] ** 2
        d = d.clamp(min=0)
        d[best] = 0

    return torch.stack(selected).sort().values


# # DPP 近似 无 SVD, 无贪心循环
def _dpp_select_optimized(
    features: torch.Tensor,
    importance: torch.Tensor,
    k: int,
    init_selected: Optional[torch.Tensor] = None,
    nms_thresh: float = 0.8,
    prefilter_ratio: float = 1.0,
) -> torch.Tensor:
    """DPP MAP 近似: 无 SVD, 无贪心循环。
    
    indep_score = imp² (L 对角, DPP greedy 第一步的精确增益)
    redundancy = 加权邻域密度 (近似条件增益衰减)
    NMS 做最终去重。
    """
    N, D = features.shape
    if k >= N:
        return torch.arange(N, device=features.device)

    feat_norm = F.normalize(features.float(), dim=-1)
    imp_norm = _td_normalize(importance.float()).clamp(min=1e-4)

    # ---- 预筛 ----
    N_keep = max(3 * k, int(N * prefilter_ratio))
    if N_keep < N:
        _, topk_idx = imp_norm.topk(N_keep)
        topk_idx = topk_idx.sort().values
        if init_selected is not None and init_selected.numel() > 0:
            topk_idx = torch.cat([topk_idx, init_selected]).unique().sort().values
            N_keep = topk_idx.shape[0]
        feat_norm = feat_norm[topk_idx]
        imp_norm = imp_norm[topk_idx]
        if init_selected is not None and init_selected.numel() > 0:
            idx_map = torch.zeros(N, dtype=torch.long, device=features.device)
            idx_map[topk_idx] = torch.arange(N_keep, device=features.device)
            init_selected = idx_map[init_selected]
    else:
        topk_idx = None

    N_sub = feat_norm.shape[0]

    # ---- 独立贡献 = L[i,i] = imp[i]² ----
    indep_score = imp_norm ** 2  # (N_sub,)

    # ---- 冗余估计: 加权邻域密度 ----
    # sim[i,j] = cos_sim(feat_i, feat_j), 对角为 0
    # redundancy[i] = Σ_j sim[i,j]² × imp[j]² / Σ_j imp[j]²
    #
    # 直觉: 如果 token i 周围有很多高 importance 且相似的 token,
    # 那它在 DPP 贪心中的条件增益会被这些邻居"抢走"。
    #
    # 但 N_sub × N_sub 的 sim 矩阵太大? 不用算完整矩阵 ——
    # 利用 feat_norm @ feat_norm.T 可以一步 matmul 出来,
    # 而且预筛后 N_sub 通常只有 100-300, 这个 matmul 很快。

    sim = feat_norm @ feat_norm.T  # (N_sub, N_sub)
    sim.fill_diagonal_(0)

    # 加权冗余: 邻居越重要 + 越相似 → 对我的条件增益削弱越大
    # 用 sim² × imp² 加权, 模拟 DPP Cholesky 中 C[step]² 的累积效应
    weighted_sim = sim ** 2 * (imp_norm ** 2).unsqueeze(0)  # (N_sub, N_sub)
    redundancy = weighted_sim.sum(dim=1) / (imp_norm ** 2).sum()  # (N_sub,) 归一化到 [0,1) 附近

    # ---- 最终评分 ----
    score = indep_score * (1.0 - redundancy)


    # # ---- 直接 Top-k ----
    # _, result = score.topk(k)


    # ---- Top 候选 + NMS ----
    n_cand = min(3 * k, N_sub)
    # n_cand = N_sub
    _, cand_idx = score.topk(n_cand)

    cand_sim = sim[cand_idx][:, cand_idx]  # 复用已算好的 sim, 不重新算

    #  # 自适应阈值: 基于候选集相似度分布
    # upper_tri = cand_sim.triu(diagonal=1)
    # upper_vals = upper_tri[upper_tri > 0]
    # if upper_vals.numel() > 0:
    #     nms_thresh = upper_vals.mean() - 1 * upper_vals.std()
    # else:
    #     nms_thresh = 0.8  # fallback

    suppress = (cand_sim > nms_thresh).triu(diagonal=1)
    is_suppressed = suppress.any(dim=0)

    survived = cand_idx[~is_suppressed]
    if survived.shape[0] >= k:
        result = survived[:k]
    else:
        suppressed = cand_idx[is_suppressed]
        result = torch.cat([survived, suppressed[:k - survived.shape[0]]])


    # ---- 合并预选 ----
    if init_selected is not None and init_selected.numel() > 0:
        mask = torch.isin(result, init_selected, invert=True)
        result_filtered = result[mask]
        n_need = k - init_selected.shape[0]
        result = torch.cat([init_selected, result_filtered[:n_need]])

    if topk_idx is not None:
        result = topk_idx[result]

    return result.sort().values


# # # # DPP 近似 有 SVD, 无贪心循环
# def _dpp_select(
#     features: torch.Tensor,
#     importance: torch.Tensor,
#     k: int,
#     init_selected: Optional[torch.Tensor] = None,
#     nms_thresh: float = 0.8,
# ) -> torch.Tensor:
#     """DPP MAP 谱松弛: svd_lowrank + NMS 离散化, 零循环。
    
#     数学背景:
#       max det(L_S) 松弛为 max det(Q^T L Q), s.t. Q^TQ=I
#       最优解 Q* = L 的 top-k 特征向量
#       token 评分 = ||Q*[i]||² = token i 在最优子空间的参与度
#       用 NMS 将连续解离散化为 k 个 token
#     """
#     N, D = features.shape
#     if k >= N:
#         return torch.arange(N, device=features.device)

#     feat_norm = F.normalize(features.float(), dim=-1)
#     imp_norm = _td_normalize(importance.float()).clamp(min=1e-4)

#     # ---- 构建 B, 使得 L = B @ B.T ----
#     B = imp_norm.unsqueeze(-1) * feat_norm  # (N, D)

#     # ---- 谱松弛: top-k 特征向量 via randomized SVD ----
#     # L 的 top-k 特征向量 = B 的 top-k 左奇异向量
#     q = min(2 * k, min(N, D))
#     U, S, _ = torch.svd_lowrank(B, q=q)  # U:(N, q), S:(q,)

#     # ---- token 评分 = 谱松弛最优解的行范数 ----
#     token_score = (U[:, :k] ** 2).sum(dim=1)  # (N,)

#     # ---- 取 top-3k 候选, NMS 离散化 ----
#     n_cand = min(3 * k, N)
#     _, cand_idx = token_score.topk(n_cand)

#     cand_sim = feat_norm[cand_idx] @ feat_norm[cand_idx].T
#     suppress = (cand_sim > nms_thresh).triu(diagonal=1)
#     is_suppressed = suppress.any(dim=0)

#     survived = cand_idx[~is_suppressed]
#     if survived.shape[0] >= k:
#         result = survived[:k]
#     else:
#         suppressed = cand_idx[is_suppressed]
#         result = torch.cat([survived, suppressed[:k - survived.shape[0]]])

#     # ---- 合并预选 ----
#     if init_selected is not None and init_selected.numel() > 0:
#         mask = torch.isin(result, init_selected, invert=True)
#         result_filtered = result[mask]
#         n_need = k - init_selected.shape[0]
#         result = torch.cat([init_selected, result_filtered[:n_need]])

#     return result.sort().values


def _facility_location_select(
    features: torch.Tensor,
    importance: torch.Tensor,
    k: int,
    init_selected: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Facility Location 贪心, (1-1/e) 保证, 减少 GPU-CPU 同步。"""
    N, D = features.shape
    if k >= N:
        return torch.arange(N, device=features.device)

    feat_norm = F.normalize(features.float(), dim=-1)
    imp = _td_normalize(importance.float()).clamp(min=1e-4)

    # 相似度矩阵
    sim = feat_norm @ feat_norm.T
    weighted_sim = sim * imp.unsqueeze(1)  # (N, N)

    # 当前覆盖值
    current_max = torch.zeros(N, device=features.device)

    # 用 tensor 存选择结果, 不回 CPU
    selected = torch.empty(k, dtype=torch.long, device=features.device)
    mask = torch.zeros(N, device=features.device, dtype=torch.bool)

    # 处理预选
    n_init = 0
    if init_selected is not None and init_selected.numel() > 0:
        n_init = init_selected.shape[0]
        for i in range(n_init):
            idx = init_selected[i]
            selected[i] = idx
            mask[idx] = True
            current_max = torch.max(current_max, weighted_sim[:, idx])

    # 贪心循环 — 全 GPU, 不调 .item()
    for step in range(n_init, k):
        # 边际增益
        marginal_gain = (weighted_sim - current_max.unsqueeze(1)).clamp(min=0).sum(dim=0)
        marginal_gain[mask] = -1.0

        # argmax 留在 GPU
        best = marginal_gain.argmax()
        selected[step] = best
        mask[best] = True

        # 更新覆盖
        current_max = torch.max(current_max, weighted_sim[:, best])

    return selected[:k].sort().values




def _facility_location_select_optimized(
    features: torch.Tensor,
    importance: torch.Tensor,
    k: int,
    init_selected: Optional[torch.Tensor] = None,
    n_greedy: int = 5,
    batch_size: int = 5,
) -> torch.Tensor:
    
    # n_greedy = k // 8
    # batch_size = k // 8

    """混合策略：前 n_greedy 个逐个贪心 + 剩余分批选。"""
    N, D = features.shape
    if k >= N:
        return torch.arange(N, device=features.device)

    feat_norm = F.normalize(features.float(), dim=-1)
    imp = _td_normalize(importance.float()).clamp(min=1e-4)

    sim = feat_norm @ feat_norm.T
    weighted_sim = sim * imp.unsqueeze(1)

    current_max = torch.zeros(N, device=features.device)
    mask = torch.zeros(N, device=features.device, dtype=torch.bool)
    selected = []

    # 处理预选
    if init_selected is not None and init_selected.numel() > 0:
        for i in range(init_selected.shape[0]):
            idx = init_selected[i]
            selected.append(idx.unsqueeze(0))
            mask[idx] = True
            current_max = torch.max(current_max, weighted_sim[:, idx])

    # ---- 阶段1：逐个贪心选 n_greedy 个种子 ----
    n_greedy = min(n_greedy, k - len(selected))
    for _ in range(n_greedy):
        marginal_gain = (weighted_sim - current_max.unsqueeze(1)).clamp(min=0).sum(dim=0)
        marginal_gain[mask] = -1.0
        best = marginal_gain.argmax()
        selected.append(best.unsqueeze(0))
        mask[best] = True
        current_max = torch.max(current_max, weighted_sim[:, best])

    # ---- 阶段2：分批选剩余 ----
    remaining = k - len(selected)
    while remaining > 0:
        batch_k = min(batch_size, remaining)

        marginal_gain = (weighted_sim - current_max.unsqueeze(1)).clamp(min=0).sum(dim=0)
        marginal_gain[mask] = -1.0

        _, batch_idx = marginal_gain.topk(batch_k)

        for i in range(batch_k):
            idx = batch_idx[i]
            selected.append(idx.unsqueeze(0))
            mask[idx] = True
            current_max = torch.max(current_max, weighted_sim[:, idx])

        remaining -= batch_k

    all_selected = torch.cat(selected)
    return all_selected[:k].sort().values




@torch.no_grad()
def _topk_trash_fuse(
    all_feats: torch.Tensor,        # (P, D)
    all_pos: torch.Tensor,          # (3, P) for M-RoPE  or  (P,) for 1D global indices
    all_imp: torch.Tensor,          # (P,)
    anchor_indices: torch.Tensor,   # (K,)
    all_frame_ids: torch.Tensor,    # (P,)
    top_k: int = 3,
    trash_ratio: float = 0.6,
    cross_frame_lambda: float = 0.0,
    fusion_temperature: float = 0.01,
    fusion_method: str = "mean",    # "mean" / "softmax" / "softmax_imp"
    anchor_weight: float = 0.5,    # mean 方法中 anchor 的权重 α, fused = α*anchor + (1-α)*drop_centroid
    return_residual_info: bool = False,  # v6: 是否返回残留池信息 (trash weights + drop data)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-K 软分配 + 自适应垃圾桶 + 向量化融合 (三种方式)。

    三种融合方式:
    - "mean": anchor-protected 均值融合, fused = α*anchor + (1-α)*drop_centroid
              drop_centroid = Σ(normalized_w × drop), anchor 始终占 α 比例, 不被稀释
    - "softmax": scatter 向量化, sim_to_centroid → per-cluster softmax → 乘 assign_weight → 归一化
    - "softmax_imp": 和 softmax 一样, 但额外乘 importance

    共同步骤:
    1. drop→anchor 相似度 (N_drop, K), 含跨帧惩罚
    2. Top-K 选择
    3. 自适应垃圾桶
    4. Softmax with trash bin → assign_weight (N_drop, K) 稀疏矩阵 P
    5. scaled_drop = assign_weight × drop_feat

    零 Python for 循环, 全向量化。

    Args:
        all_feats: (P, D) 所有 token 特征 (anchor + drop)
        all_pos: (3, P) M-RoPE 位置 (Qwen2.5/3-VL) 或 (P,) 全局索引 (LLaVA)
        all_imp: (P,) 所有 token 的 importance 分数
        anchor_indices: (K,) anchor token 的索引
        all_frame_ids: (P,) 每个 token 的 local frame id
        top_k: 每个 drop token 分配到的 anchor 数量
        trash_ratio: 自适应垃圾桶阈值比例 (0 = 无垃圾桶)
        cross_frame_lambda: 跨帧时间距离惩罚系数
        fusion_temperature: softmax 温度参数
        fusion_method: 融合方式 ("mean" / "softmax" / "softmax_imp")
        anchor_weight: mean 方法中 anchor 的权重 α ∈ (0,1], 默认 0.5
                       fused = α * anchor + (1-α) * drop_centroid
                       α=0.5 → anchor 贡献 50%, α=1.0 → 纯 anchor (不融合)

    Returns:
        fused_feats: (K, D) 融合后的 anchor 特征
        fused_pos: (3, K) 或 (K,) anchor 位置 (不变, 维度与 all_pos 一致)
        residual_info: (仅 return_residual_info=True) dict 或 None, 含:
            drop_feats (N_drop, D), trash_weights (N_drop,), drop_pos ((3, N_drop) 或 (N_drop,)),
            drop_imp (N_drop,), drop_frame_ids (N_drop,), drop_mask_indices (N_drop,), count
    """
    P, D = all_feats.shape
    K = anchor_indices.shape[0]
    device = all_feats.device

    # 构建 anchor / drop mask
    is_anchor = torch.zeros(P, dtype=torch.bool, device=device)
    is_anchor[anchor_indices] = True

    anchor_feats = all_feats[anchor_indices].float()   # (K, D)
    # 支持 (3, P) M-RoPE 和 (P,) 全局索引两种 position 格式
    _pos_is_1d = (all_pos.ndim == 1)
    anchor_pos = all_pos[anchor_indices] if _pos_is_1d else all_pos[:, anchor_indices]
    anchor_frame_ids = all_frame_ids[anchor_indices]    # (K,)

    drop_mask = ~is_anchor
    drop_mask_indices = drop_mask.nonzero(as_tuple=True)[0]  # (N_drop,)
    N_drop = drop_mask_indices.shape[0]

    if N_drop == 0:
        if return_residual_info:
            return anchor_feats.to(all_feats.dtype), anchor_pos, None

    # ---- anchor_weight >= 1.0: 短路 ----
    # anchor 完全不吸收 drop token, 跳过 softmax 分配
    # 所有 drop token 以原始特征 / 原始 importance 进入残差池
    if anchor_weight >= 1.0:
        fused = anchor_feats
        if return_residual_info:
            residual_dict = {
                'drop_feats': all_feats[drop_mask_indices].float(),
                'trash_weights': torch.ones(N_drop, device=device),
                'drop_pos': all_pos[drop_mask_indices] if _pos_is_1d else all_pos[:, drop_mask_indices],
                'drop_imp': all_imp[drop_mask_indices].float(),
                'drop_frame_ids': all_frame_ids[drop_mask_indices],
                'drop_mask_indices': drop_mask_indices,
                'count': N_drop,
            }
            return fused.to(all_feats.dtype), anchor_pos, residual_dict
        return fused.to(all_feats.dtype), anchor_pos


    drop_feats = all_feats[drop_mask_indices].float()        # (N_drop, D)
    drop_imp = all_imp[drop_mask_indices].float()             # (N_drop,)
    drop_frame_ids = all_frame_ids[drop_mask_indices]         # (N_drop,)

    # ---- Step 1: drop → anchor 相似度 ----
    drop_norm = F.normalize(drop_feats, dim=-1)               # (N_drop, D)
    anchor_norm = F.normalize(anchor_feats, dim=-1)           # (K, D)
    sims = drop_norm @ anchor_norm.T                          # (N_drop, K)

    # 跨帧惩罚
    if cross_frame_lambda == float('inf'):
        frame_match = (drop_frame_ids.unsqueeze(1) == anchor_frame_ids.unsqueeze(0))
        sims[~frame_match] = -float('inf')
    elif cross_frame_lambda > 0:
        frame_dist = (drop_frame_ids.unsqueeze(1) - anchor_frame_ids.unsqueeze(0)).abs().float()
        sims = sims - cross_frame_lambda * frame_dist

    # ---- Step 2: Top-K 选择 ----
    actual_k = min(top_k, K)
    topk_sims, topk_ids = sims.topk(actual_k, dim=1)         # (N_drop, actual_k)

    # ---- Step 3: 自适应垃圾桶 ----
    if trash_ratio > 0 and N_drop > 1:
        # 两段式: N_drop 小则一次性精确算, 大则用 10% 子采样近似
        # 数学语义等价于 (drop_norm @ drop_norm.T).fill_diagonal_(0).max(dim=1)
        if N_drop <= 10000:
            # 精确路径: (N_drop, N_drop) ≤ 64 MB, 一次跑完省 kernel launch
            sim_full = drop_norm @ drop_norm.T            # (N_drop, N_drop)
            sim_full.fill_diagonal_(0)
            per_token_max_sim = sim_full.max(dim=1).values

            # # 子采样近似: 每个 drop 只看 N_drop/10 个随机邻居取 max
            # # max 为单调统计量, 子采样只会低估而不会高估,
            # # 偏低门槛 → 更多 token 进 anchor (信息保留更多, 方向安全)
            # m = N_drop // 4
            # rand_idx = torch.randint(0, N_drop, (m,), device=device)
            # sim_sub = drop_norm @ drop_norm[rand_idx].T   # (N_drop, m)
            # # 排除自身碰巧被采到的位置
            # self_mask = (torch.arange(N_drop, device=device).unsqueeze(1)
            #              == rand_idx.unsqueeze(0))
            # sim_sub.masked_fill_(self_mask, 0.0)
            # per_token_max_sim = sim_sub.max(dim=1).values

        else:
            # 子采样近似: 每个 drop 只看 N_drop/10 个随机邻居取 max
            # max 为单调统计量, 子采样只会低估而不会高估,
            # 偏低门槛 → 更多 token 进 anchor (信息保留更多, 方向安全)
            m = N_drop // 5
            rand_idx = torch.randint(0, N_drop, (m,), device=device)
            sim_sub = drop_norm @ drop_norm[rand_idx].T   # (N_drop, m)
            # 排除自身碰巧被采到的位置
            self_mask = (torch.arange(N_drop, device=device).unsqueeze(1)
                         == rand_idx.unsqueeze(0))
            sim_sub.masked_fill_(self_mask, 0.0)
            per_token_max_sim = sim_sub.max(dim=1).values

        # # 考虑drop和drop相似度
        # trash_threshold = (per_token_max_sim * trash_ratio).unsqueeze(1)  # (N_drop, 1)

        # 不仅考虑drop和drop相似度还考虑drop和anchor的不相似度
        anchor_deficit = 1.0 - topk_sims[:, 0:1]                         # (N_drop, 1)
        trash_threshold = torch.max(
            (per_token_max_sim * trash_ratio).unsqueeze(1),
            anchor_deficit * trash_ratio
        )                                                                  # (N_drop, 1)

    else:
        trash_threshold = torch.zeros(N_drop, 1, device=device)

    # ---- Step 4: Softmax with trash bin → 分配权重矩阵 P ----
    sims_with_trash = torch.cat([topk_sims / fusion_temperature,
                                  trash_threshold / fusion_temperature], dim=1)
    raw_weights = F.softmax(sims_with_trash, dim=1)           # (N_drop, actual_k+1)
    drop_anchor_weights = raw_weights[:, :actual_k]           # (N_drop, actual_k)
    # raw_weights[:, -1] 是垃圾桶权重 (丢弃的部分)

    # 构建 (N_drop, K) 稀疏分配矩阵 P_mat
    P_mat = torch.zeros(N_drop, K, device=device, dtype=torch.float32)
    P_mat.scatter_(1, topk_ids, drop_anchor_weights)

    # ---- 残留池权重 (v6: residual pool recovery) ----
    _trash_weights_for_residual = raw_weights[:, -1]  # (N_drop,) 每个 drop 的垃圾桶权重

    # ---- Step 5: 缩放 drop 特征 ----
    # scaled_drop_feats[i] = sum_j P_mat[i,j] * drop_feats[i] 的概念
    # 但实际上每个 drop 只需乘以其对各 anchor 的权重
    # 对于 mean 方式: weighted_drop = P_mat.T @ drop_feats 已自带缩放
    # 对于 softmax 方式: 需要单独的 assign_weight

    # ========================================================================
    # 融合方式 1: mean — anchor-protected 均值融合
    #   fused = α * anchor + (1-α) * drop_centroid
    #   drop_centroid = Σ(normalized_w × drop), 其中 normalized_w = w_i / Σw_j
    #   anchor 始终占 α 比例, 不受簇大小影响, 防止被大量 drop 稀释
    # ========================================================================
    if fusion_method == "mean":
        # P_mat.T @ drop_feats: 每个 anchor 累加其分配到的 drop 特征 (已乘权重)
        weighted_drop = P_mat.T @ drop_feats                  # (K, D)
        # 每个 anchor 的 drop 权重总和
        drop_weight_sum = P_mat.sum(dim=0)                    # (K,)
        # 归一化: drop_centroid = weighted_drop / sum_of_weights
        # 如果某个 anchor 没有 drop 分配, 则 centroid = 0, 公式退化为 fused = anchor
        safe_sum = drop_weight_sum.clamp(min=1e-8)
        drop_centroid = weighted_drop / safe_sum.unsqueeze(-1)  # (K, D)
        # 有 drop 的 anchor: fused = α * anchor + (1-α) * drop_centroid
        # 无 drop 的 anchor: fused = anchor (不变)
        has_drop = (drop_weight_sum > 1e-8).float().unsqueeze(-1)  # (K, 1)
        fused = has_drop * (anchor_weight * anchor_feats + (1.0 - anchor_weight) * drop_centroid) + (1.0 - has_drop) * anchor_feats

        if return_residual_info:
            residual_dict = {
                'drop_feats': drop_feats,
                'trash_weights': _trash_weights_for_residual,
                'drop_pos': all_pos[drop_mask_indices] if _pos_is_1d else all_pos[:, drop_mask_indices],
                'drop_imp': drop_imp,
                'drop_frame_ids': all_frame_ids[drop_mask_indices],
                'drop_mask_indices': drop_mask_indices,
                'count': N_drop,
            }
            return fused.to(all_feats.dtype), anchor_pos, residual_dict
        return fused.to(all_feats.dtype), anchor_pos

    # ========================================================================
    # 融合方式 2 & 3: softmax / softmax_imp — scatter 向量化
    # ========================================================================
    # 共同步骤: argmax 硬分配, 用 anchor 自身作为 centroid (省去 scatter_mean),
    #   anchor 天然 cos_sim=1.0, 获得隐式保护

    # 1. argmax 硬分配: drop -> best anchor
    # 优化 #4: topk 的第一个就是 argmax, 复用 topk_ids[:, 0] 省一次 max(dim=1)
    best_anchor = topk_ids[:, 0]                              # (N_drop,)

    full_assign = torch.empty(P, dtype=torch.long, device=device)
    full_assign[anchor_indices] = torch.arange(K, device=device)
    full_assign[drop_mask_indices] = best_anchor

    # 2. centroid = anchor_feats (直接用 anchor 作为簇中心, 无需 scatter_mean)
    full_feats = all_feats.float()                            # (P, D)

    # 3. sim = cos_sim(token, its anchor centroid)
    # 优化 #1 + #6: anchor 自身的 cos_sim 恒为 1.0, 无需计算; drop 复用 Step 1 的
    # drop_norm / anchor_norm 直接点积 (避免 cosine_similarity 内部再次 normalize)
    sim = torch.ones(P, device=device, dtype=drop_norm.dtype)
    # drop_norm: (N_drop, D), anchor_norm[best_anchor]: (N_drop, D)
    drop_sim = (drop_norm * anchor_norm[best_anchor]).sum(dim=-1)   # (N_drop,)
    sim[drop_mask_indices] = drop_sim

    # 4. per-cluster softmax(sim / T)
    raw_scaled = sim / fusion_temperature
    cluster_max = torch.full((K,), -float("inf"), device=device)
    cluster_max.scatter_reduce_(0, full_assign, raw_scaled, reduce="amax", include_self=False)
    raw_shifted = raw_scaled - cluster_max[full_assign]
    exp_w = torch.exp(raw_shifted)
    sum_exp = torch.zeros(K, device=device)
    sum_exp.scatter_add_(0, full_assign, exp_w)
    sim_weights = exp_w / sum_exp[full_assign].clamp(min=1e-8)  # (P,)

    # 5. 对 drop token 乘以 assign_weight (Top-K 贡献), anchor 的 assign_weight = 1
    # 优化 #4: best_anchor = topk_ids[:, 0], 所以分配权重直接是 drop_anchor_weights[:, 0]
    # 省掉 P_mat[arange, best_anchor] 这次 gather
    token_assign_w = torch.ones(P, device=device)
    token_assign_w[drop_mask_indices] = drop_anchor_weights[:, 0]

    if fusion_method == "softmax_imp":
        # softmax_imp: 额外乘 importance
        combined_w = sim_weights * token_assign_w * all_imp.float()
    else:
        # softmax: 不乘 importance
        combined_w = sim_weights * token_assign_w

    # 6. per-cluster 归一化
    w_sum = torch.zeros(K, device=device)
    w_sum.scatter_add_(0, full_assign, combined_w)
    final_w = combined_w / w_sum[full_assign].clamp(min=1e-8)  # (P,)

    # 7. scatter 融合
    fused = torch.zeros(K, D, device=device, dtype=torch.float32)
    fused.scatter_add_(0, full_assign.unsqueeze(1).expand(P, D),
                       final_w.unsqueeze(1) * full_feats)

    if return_residual_info:
        residual_dict = {
            'drop_feats': drop_feats,
            'trash_weights': _trash_weights_for_residual,
            'drop_pos': all_pos[drop_mask_indices] if _pos_is_1d else all_pos[:, drop_mask_indices],
            'drop_imp': drop_imp,
            'drop_frame_ids': all_frame_ids[drop_mask_indices],
            'drop_mask_indices': drop_mask_indices,
            'count': N_drop,
        }
        return fused.to(all_feats.dtype), anchor_pos, residual_dict
    return fused.to(all_feats.dtype), anchor_pos








@torch.no_grad()
def _dpc_knn_cluster(
    features: Tensor,
    num_clusters: int,
    k: int = 7,
    importance: Tensor = None,
) -> Tuple[Tensor, Tensor]:
    """
    DPC-kNN clustering — optimized version.
    Improvements:
      - cosine distance via matmul (faster than cdist)
      - vectorized delta computation (no Python loop)
      - importance-weighted center selection & cluster fusion
    """
    N, D = features.shape
    if num_clusters >= N:
        return features, torch.arange(N, device=features.device)

    # === Step 1: pairwise cosine distance via matmul (优化 #5) ===============
    # 比 torch.cdist 快 ~2x，利用 cuBLAS GEMM
    features_f = features.float()
    features_norm = F.normalize(features_f, p=2, dim=-1)
    dist = 1.0 - (features_norm @ features_norm.T)          # [N, N]

    # === Step 2: k-NN density ================================================
    dist_topk, _ = dist.topk(k=k + 1, dim=-1, largest=False)   # include self
    knn_avg_dist = dist_topk[:, 1:].mean(dim=-1)               # exclude self
    density = 1.0 / (knn_avg_dist + 1e-8)

    # === Step 3: delta — 全向量化 (优化 #6) ==================================
    # 原始: Python for loop O(N^2)，现在: 纯 tensor 操作，GPU 并行
    # density_mask[i,j] = True  ⟺  density[j] > density[i]
    density_mask = density.unsqueeze(0) > density.unsqueeze(1)  # [N, N]
    masked_dist = dist.masked_fill(~density_mask, float('inf'))
    delta, _ = masked_dist.min(dim=1)                           # [N]
    # 最高密度点没有更高密度邻居 → 赋值为其他点的 max delta
    max_density_idx = density.argmax()
    valid_delta = delta[delta < float('inf')]
    if valid_delta.numel() > 0:
        delta[max_density_idx] = valid_delta.max()
    else:
        delta[max_density_idx] = 1.0

    # === Step 4: center selection — importance 加权 (优化 #3) =================
    if importance is not None:
        _imp_norm = importance.float()
        _imp_min, _imp_max = _imp_norm.min(), _imp_norm.max()
        if (_imp_max - _imp_min) > 1e-8:
            _imp_norm = (_imp_norm - _imp_min) / (_imp_max - _imp_min)
        else:
            _imp_norm = torch.ones_like(_imp_norm)
        _imp_norm = _imp_norm.clamp(min=0.1)
        score = density * delta + _imp_norm
        # _td_logger.info(f"[DEBUG] score min: {score.min():.4f}, max: {score.max():.4f}, _imp_norm min: {_imp_norm.min():.4f}, max: {_imp_norm.max():.4f}")
    else:
        score = density * delta
    center_indices = torch.topk(score, k=num_clusters, dim=-1).indices

    # === Step 5: assign each token to nearest center =========================
    # 优化 #A: 复用 Step 1 的 dist, 切片即可, 省 (N,D)@(D,C) ≈ N·C·D FLOPs
    # 等价于 features_norm @ features_norm[center_indices].T  (因为 dist=1-sim)
    sim_to_centers = 1.0 - dist[:, center_indices]              # [N, C]
    assignments = sim_to_centers.argmax(dim=-1)                 # [N]

    # === Step 6: importance-weighted cluster fusion ===========================
    # 优化 #F: 改用 scatter_add_, 避免 (N, C) one_hot 物化 + (C,N)@(N,D) 稠密 matmul
    # 旧 FLOPs: C·N·D ; 新 FLOPs: N·D + N (scatter_add) ≈ 降 ~C 倍
    C = num_clusters
    if importance is not None:
        weights = importance.float().clamp(min=1e-8)
    else:
        weights = torch.ones(N, device=features.device, dtype=torch.float32)

    weighted_feats = features_f * weights.unsqueeze(-1)         # [N, D]
    cluster_feats = torch.zeros(C, D, device=features.device, dtype=torch.float32)
    cluster_feats.scatter_add_(
        0,
        assignments.unsqueeze(-1).expand(N, D),
        weighted_feats,
    )
    weight_sums = torch.zeros(C, device=features.device, dtype=torch.float32)
    weight_sums.scatter_add_(0, assignments, weights)
    cluster_feats = cluster_feats / weight_sums.clamp(min=1e-8).unsqueeze(-1)

    return cluster_feats, center_indices



def _dyseg_group_frames(frame_features_norm: List[torch.Tensor],
                        threshold: float,
                        min_segment_num: int = 0,
                        complementary_segment: bool = True) -> List[List[int]]:
    """DySeg: 将相邻帧按余弦相似度分组, 支持最少分段数约束。

    相邻帧 mean feature 的 cosine similarity > threshold → 同组。
    若分组数 < min_segment_num 且 complementary_segment=True,
    则在剩余位置中选相似度最低的补刀, 直到满足最少分段数。

    Args:
        frame_features_norm: List of (K_t, D) normalized frame features.
        threshold: 相似度阈值, 越高分组越细。
        min_segment_num: 最少分段数, 0 表示不限制。
        complementary_segment: 不够 min_segment_num 时是否自动补刀。

    Returns:
        groups: List of List[int], 每组包含的帧索引。
    """
    T = len(frame_features_norm)
    if T <= 1:
        return [list(range(T))]

    # 计算每帧的 mean feature
    frame_means = torch.stack([f.mean(dim=0) for f in frame_features_norm])  # (T, D)
    frame_means = F.normalize(frame_means, dim=-1)

    # 相邻帧相似度 (T-1,)
    transition_sims = F.cosine_similarity(frame_means[:-1], frame_means[1:], dim=-1)
    transition_sims = _td_normalize(transition_sims)

    # Step 1: 阈值切分 — 相似度 < threshold 的位置切一刀
    cut_indices = (transition_sims < threshold).nonzero(as_tuple=True)[0].tolist()
    
    _td_logger.info(f"[DySeg] T={T}, threshold={threshold}, natural_cuts={len(cut_indices)+1}, sims: min={transition_sims.min():.4f}, max={transition_sims.max():.4f}, mean={transition_sims.mean():.4f}")

    # Step 2: 补刀 — 分段数不够 min_segment_num 时, 选剩余最低相似度位置
    num_segments = len(cut_indices) + 1
    if min_segment_num > 0 and num_segments < min_segment_num and complementary_segment:
        num_needed = min_segment_num - num_segments
        # 把已切位置的相似度设为 1.0 (排除)
        remaining_sims = transition_sims.clone()
        for idx in cut_indices:
            remaining_sims[idx] = 1.0
        # 从剩余位置中选相似度最低的 top-K
        k = min(num_needed, remaining_sims.shape[0])
        if k > 0:
            extra_indices = torch.topk(remaining_sims, k=k, largest=False).indices.tolist()
            cut_indices = sorted(set(cut_indices + extra_indices))

    # Step 3: 根据切点构建分组
    cut_indices_sorted = sorted(cut_indices)
    groups = []
    prev = 0
    for c in cut_indices_sorted:
        # cut at position c means: frames [prev..c] are one group, [c+1..] starts next
        groups.append(list(range(prev, c + 1)))
        prev = c + 1
    groups.append(list(range(prev, T)))

    return groups


def _compute_group_budget(
    groups: List[List[int]],
    frame_features_norm: List[torch.Tensor],  # list of (N, D) normalized
    total_budget: int,
    method: str,  # "effective_rank", "effective_rank2", "uniform", "mcd"
    N_real: List[int],
    device=None,
    token_importance_per_frame: List[torch.Tensor] = None,
    effective_rank_k: int = 64,
) -> List[int]:
    """计算每组的 token 预算。

    支持 4 种方法:

    Effective Rank (有效秩, 基于 eigvalsh 的全谱):
      Gram 矩阵 → eigvalsh → 归一化奇异值 → exp(Shannon entropy)
      衡量"需要多少个独立方向来表示这个组的信息"。
      已隐式包含组大小信息: 更大的组如果内容多样 → 更高的有效秩。
      因此无需额外乘以 len(group)。

    Effective Rank 2 (有效秩, 基于 svd_lowrank 的截断谱):
      randomized SVD (q ≈ k·sqrt(len(group))) → 归一化奇异值 → exp(Shannon entropy)
      与 effective_rank 同语义,但用 svd_lowrank 加速,适合大组。

    Uniform (均匀):
      group_weights = P,即按组的 token 总数分配,等同于 per-token 均分预算。

    MCD (Mean Cosine Distance):
      1 - mean(pairwise_sim)
      衡量组内 token 的平均差异度。
      是 scale-insensitive 的, 需要乘以 len(group) 来反映组大小。

    Args:
        groups: DySeg 分组结果
        frame_features_norm: 每帧的归一化特征
        total_budget: 总 token 预算
        method: "effective_rank", "effective_rank2", "uniform", "mcd"
        N_real: 每帧 token 数
        device: 计算设备

    Returns:
        group_budgets: 每组的 token 预算 (整数列表, 和 = total_budget)
    """
    num_groups = len(groups)
    group_weights = torch.zeros(num_groups, device=device)

    for g_idx, group in enumerate(groups):
        # 收集该组所有 token
        feats_list = []
        for t in group:
            feats_list.append(frame_features_norm[t])
        pool_feats = torch.cat(feats_list, dim=0)  # (P, D)
        P = pool_feats.shape[0]

        if P <= 1:
            group_weights[g_idx] = 1.0
            continue

        if method == "effective_rank":
            # SVD → 归一化奇异值 → exp(Shannon entropy)
            # 有效秩直接代表信息维度数, 不需要额外缩放
            # S = torch.linalg.svdvals(pool_feats.float())


            # # 计算有效秩
            # P, D = pool_feats.shape
            # k = 16
            # q = int(min(k * math.sqrt(len(group)), P, D))
            # _, S, _ = torch.svd_lowrank(pool_feats.float(), q=q, niter=2)  # randomized SVD, ~56x faster


            # # 用协方差矩阵特征值代替 SVD
            # gram = pool_feats.float().T @ pool_feats.float()  # (D, D)
            # eigvals = torch.linalg.eigvalsh(gram)              # (D,) 升序, 精确
            # S = torch.sqrt(eigvals.clamp(min=1e-8))

            # # 同样使用协方差矩阵代替SVD 但是只取前P_sub个特征值用以加速
            # P_sub = min(1024, P)
            # if P > P_sub:
            #     idx = torch.randperm(P, device=device)[:P_sub]
            #     pool_sub = pool_feats[idx]
            # else:
            #     pool_sub = pool_feats
            # gram = pool_sub.float().T @ pool_sub.float()  # (D, D)
            # eigvals = torch.linalg.eigvalsh(gram)          # (D,) 升序, 精确
            # S = torch.sqrt(eigvals.clamp(min=1e-8))


            # 加速之使用小值
            P, D = pool_feats.shape
            feats = pool_feats.float()

            if P < D:
                gram = feats @ feats.T                        # (P, P) 而不是 (D, D)
            else:
                gram = feats.T @ feats                        # (D, D)

            eigvals = torch.linalg.eigvalsh(gram)
            S = torch.sqrt(eigvals.clamp(min=0))
            
            
            S = S[S > 1e-4]  # 过滤掉真正的噪声
            if S.numel() == 0:
                group_weights[g_idx] = 1.0
                continue
            p = S / S.sum()
            eff_rank = torch.exp(-(p * torch.log(p + 1e-8)).sum())
            
            # 混合 uniform 保底
            alpha = 0.05
            group_weights[g_idx] =  (1 - alpha) *  eff_rank.item() + alpha * P
        
        
        elif method == "effective_rank2":
            # 计算有效秩
            P, D = pool_feats.shape
            k = max(1, int(effective_rank_k))
            q = int(min(k * math.sqrt(len(group)), P, D))
            _, S, _ = torch.svd_lowrank(pool_feats.float(), q=q, niter=2)  # randomized SVD, ~56x faster
            S = S[S > 1e-4]  # 过滤掉真正的噪声
            if S.numel() == 0:
                group_weights[g_idx] = 1.0
                continue
            p = S / S.sum()
            eff_rank = torch.exp(-(p * torch.log(p + 1e-8)).sum())
            
            # 混合 uniform 保底
            alpha = 0.05
            group_weights[g_idx] = (1 - alpha) * eff_rank.item() + alpha * P

        elif method == "uniform":
            group_weights[g_idx] = P

        elif method == "mcd":
            # Mean Cosine Distance: 1 - mean(pairwise_sim)
            # 需要乘以组大小, 因为 MCD 本身是 scale-insensitive 的
            sim_matrix = pool_feats @ pool_feats.T  # (P, P)
            mask_ut = torch.triu(torch.ones(P, P, device=device, dtype=torch.bool), diagonal=1)
            mean_sim = sim_matrix[mask_ut].mean() if mask_ut.any() else torch.tensor(0.5, device=device)
            mcd = (1.0 - mean_sim.item())
            group_weights[g_idx] = mcd * P  # 乘以组大小

        else:
            raise NotImplementedError(
                f"group_budget method '{method}' is not supported. "
                f"Only 'effective_rank' / 'effective_rank2' / 'uniform' / 'mcd' are supported."
            )


    # 归一化并分配预算
    if group_weights.sum() <= 0:
        group_weights = torch.ones(num_groups, device=device)
    ratio = group_weights / group_weights.sum()
    group_budgets_float = ratio * total_budget
    group_budgets = group_budgets_float.floor().int().tolist()

    # 确保每组至少 1 个 token
    group_budgets = [max(1, b) for b in group_budgets]
    # 限制每组预算不超过其实际 token 数
    for g_idx, group in enumerate(groups):
        max_tokens = sum(N_real[t] for t in group)
        group_budgets[g_idx] = min(group_budgets[g_idx], max_tokens)

    # 分配余量 (按权重降序)
    remainder = total_budget - sum(group_budgets)
    sorted_groups = sorted(range(num_groups), key=lambda i: -group_weights[i].item())
    idx = 0
    while remainder > 0 and idx < num_groups * 10:
        g = sorted_groups[idx % num_groups]
        max_tokens = sum(N_real[t] for t in groups[g])
        if group_budgets[g] < max_tokens:
            group_budgets[g] += 1
            remainder -= 1
        idx += 1

    # 修正超额
    if sum(group_budgets) > total_budget:
        sorted_asc = sorted(range(num_groups), key=lambda i: group_weights[i].item())
        excess = sum(group_budgets) - total_budget
        idx = 0
        while excess > 0 and idx < num_groups * 10:
            g = sorted_asc[idx % num_groups]
            if group_budgets[g] > 1:
                group_budgets[g] -= 1
                excess -= 1
            idx += 1

    return group_budgets


# ============================================================================
# 主压缩函数: DySeg + Group DPP + Top-K Soft Fusion
# ============================================================================


@torch.no_grad()
def tensor_decomp_compression_qwen(
    video_embeds: torch.Tensor,
    position_ids_video: torch.Tensor,
    num_frames: int,
    tokens_per_frame: int,
    config,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Post-projector 压缩 Pipeline:
      DySeg分组 → 组预算分配(Effective Rank/MCD) → 组内DPP anchor选择 → Top-K软融合

    所有信号 (CLS attn, dynamism) 在此处计算 (projector 之后)。
    ViT 完全不修改。多样性由 DPP kernel 自动处理, 无需 uniqueness 信号。

    Args:
        video_embeds: (T * N, D) 所有视频 token (projector 输出)。
        position_ids_video: (3, T * N) M-RoPE 位置 ID。
        num_frames: T。
        tokens_per_frame: N。
        config: TensorDecompConfig 实例。

    Returns:
        compressed_embeds: (budget, D) 压缩后的视频 token。
        compressed_positions: (3, budget) 压缩后的位置 ID。
        kept_global_indices: (budget,) 每个保留 token 在原始 flat 视觉序列中的全局索引。
    """
    T = num_frames
    N = tokens_per_frame
    D = video_embeds.shape[-1]
    device = video_embeds.device
    dtype = video_embeds.dtype

    N_original = getattr(config, '_original_tokens_per_frame', None) or N
    _s2_ratio = getattr(config, 'stage2_retention_ratio', 0.20) or 0.20
    budget = max(1, int(T * N_original * _s2_ratio))

    # 如果不需要压缩
    if budget >= T * N:
        # _td_logger.info(f"[Compress] SKIP: budget={budget} >= T*N={T*N}")
        return video_embeds, position_ids_video, torch.arange(T * N, device=device)

    # ---- 拆分为逐帧 ----
    frame_features = video_embeds.view(T, N, D)
    frame_positions = position_ids_video.view(3, T, N)

    frame_features_list = [frame_features[t] for t in range(T)]       # list of (N, D)
    frame_positions_list = [frame_positions[:, t, :] for t in range(T)]  # list of (3, N)

    # ---- 读取新版配置参数 ----
    dynamism_window = config.dynamism_window
    w_dyn = config.w_dyn                       # dynamism 权重 (convex combination)
    asym_w = config.asym_w                     # dynamism 中 asymmetry 的权重
    fusion_temperature = config.fusion_temperature
    cross_frame_lambda = config.cross_frame_lambda
    dyseg_threshold = config.dyseg_threshold
    min_segment_num = getattr(config, 'min_segment_num', 0)           # 最少分段数 (0=不限制)
    complementary_segment = getattr(config, 'complementary_segment', True)  # 不够时自动补刀
    topk_fusion = config.topk_fusion           # Top-K 软分配的 K
    trash_ratio = config.trash_ratio           # 自适应垃圾桶比例
    fusion_method = getattr(config, 'fusion_method', 'mean')
    anchor_weight = getattr(config, 'anchor_weight', 0.5)
    res_imp_alpha = getattr(config, 'res_imp_alpha', 0.3)   # DPP₂ importance 压缩指数 (anchor_weight>=1.0 时生效)
    res_method = getattr(config, 'res_method', 'dpc_knn')         # 残差池处理方式: "dpp" 或 "dpc_knn"
    anchor_method = getattr(config, 'anchor_method', 'dpp')
    residual_ratio = getattr(config, 'residual_ratio', 0.0)
    min_compression_ratio = getattr(config, 'min_compression_ratio', 3)  # 先选后替: trash_pool >= K_replace * ratio 才替换
    group_budget_method = config.group_budget_method  # "effective_rank" or "mcd"
    min_tokens_per_frame = 1  # 每帧最少保留 1 个 anchor

    # Normalize features for similarity computation
    frame_features_norm = [F.normalize(f.float(), dim=-1) for f in frame_features_list]

    # Signal 1: ViT col_mean CLS attention (产线只支持 col_mean)
    _cls_method = getattr(config, 'cls_attn_method', 'col_mean')
    if _cls_method != 'col_mean':
        raise NotImplementedError(
            f"Only cls_attn_method='col_mean' is supported, got '{_cls_method}'"
        )
    _vit_cls_attn = getattr(config, '_vit_cls_attn', None)
    assert _vit_cls_attn is not None and _vit_cls_attn.shape[0] == T * N, (
        f"col_mean signal not available: shape="
        f"{_vit_cls_attn.shape if _vit_cls_attn is not None else None}, expected=({T*N},)"
    )
    token_norms_per_frame = list(_vit_cls_attn.float().view(T, N))  # list of (N,)
    # _td_logger.info(
    #     f"[Compress] Signal 1: ViT col_mean attention "
    #     f"(range [{_vit_cls_attn.min().item():.4f}, {_vit_cls_attn.max().item():.4f}])"
    # )

    # Signal 2: Dynamism (新公式: (1-asym_w)*novelty + asym_w*asymmetry)
    # novelty = 1 - max(fwd_max, bwd_max)  → 前后帧都不相似 = 新内容
    # asymmetry = |fwd_max - bwd_max|       → 前后不对称 = 运动方向变化
    # [优化] sim 矩阵复用: 每对相邻帧 (t, t+d) 只做一次 (N,N) matmul,
    #   max(dim=1) → fwd_max[t], max(dim=0) → bwd_max[t+d]
    # [批量化] 对每个固定 d, 所有 t 互相独立 → bmm 一次出 (T-d, N, N)
    w = dynamism_window

    fwd_max_all = torch.zeros(T, N, device=device, dtype=torch.float32)
    bwd_max_all = torch.zeros(T, N, device=device, dtype=torch.float32)

    if T > 1:
        # 把 list 堆成 (T, N, D), 注意 frame_features_norm[t] 已经是 float32
        frames_stack = torch.stack(frame_features_norm, dim=0)  # (T, N, D)

        for d in range(1, min(w, T - 1) + 1):
            # 对所有 t in [0, T-d): 计算 frames[t] @ frames[t+d].T
            sim_batch = torch.bmm(
                frames_stack[: T - d],                       # (T-d, N, D)
                frames_stack[d:].transpose(-2, -1),          # (T-d, D, N)
            )  # (T-d, N, N)
            fwd_d = sim_batch.max(dim=2).values              # (T-d, N)
            bwd_d = sim_batch.max(dim=1).values              # (T-d, N)
            fwd_max_all[: T - d] = torch.max(fwd_max_all[: T - d], fwd_d)
            bwd_max_all[d:]      = torch.max(bwd_max_all[d:],      bwd_d)

    # 一次性按公式算 dynamism: novelty = 1 - max(fwd, bwd), asymmetry = |fwd - bwd|
    novelty_all   = (1.0 - torch.max(fwd_max_all, bwd_max_all)).clamp(min=0)  # (T, N)
    asymmetry_all = torch.abs(fwd_max_all - bwd_max_all)                       # (T, N)
    token_dyn_all = (1.0 - asym_w) * novelty_all + asym_w * asymmetry_all      # (T, N)
    token_dynamism_per_frame = list(token_dyn_all)  # list of (N,) views

    # Normalize signals
    # dynamism: 保留原始值, 归一化移到组内进行 (跨帧绝对值可比)
    token_norms_per_frame = [_td_normalize(tn) for tn in token_norms_per_frame]

    # _td_logger.info(
    #     f"[Compress] Signals (T={T}, N={N}): "
    #     f"dynamism formula: (1-{asym_w})*novelty + {asym_w}*asymmetry"
    # )

    # ---- DySeg 分组 ----
    groups = _dyseg_group_frames(frame_features_norm, dyseg_threshold, min_segment_num, complementary_segment)
    _td_logger.info(
        f"[Compress] DySeg: threshold={dyseg_threshold}, "
        f"groups={len(groups)}, sizes={[len(g) for g in groups]}"
    )

    # 每帧 token 数 (无 padding, 所有帧等长)
    _N_real = [N] * T

    # ---- 预计算每帧 token importance ----
    # cls_attn: 已帧内归一化 (帧内相对排序, 帧间不可比)
    # dynamism: 保留原始值, 组内归一化 (跨帧绝对值可比)
    # 最终 importance = convex combination, 在组循环内计算
    w_cls = max(0.0, 1.0 - w_dyn)  # cls_attn 的权重

    token_imp_per_frame = []  # 帧级暂存: 仅 cls 分量 (已归一化)
    for t in range(T):
        _t_norms = token_norms_per_frame[t]
        _t_dyn = token_dynamism_per_frame[t]   # 原始值 (未归一化)

        # _td_logger.info(
        #     f"[Compress] DPP signals (T={t}): cls=({_t_norms.min().item():.3f},{_t_norms.max().item():.3f}) dyn_raw=({_t_dyn.min().item():.3f},{_t_dyn.max().item():.3f})"
        # )

        # 帧级只暂存 cls 分量
        token_imp_per_frame.append(_t_norms)

    # ---- 组预算分配 ----
    group_budgets = _compute_group_budget(
        groups=groups,
        frame_features_norm=frame_features_norm,
        total_budget=budget,
        method=group_budget_method,
        N_real=_N_real,
        device=device,
        token_importance_per_frame=token_norms_per_frame,
        effective_rank_k=int(getattr(config, 'effective_rank_k', 64)),
    )

    # _td_logger.info(
    #     f"[Compress] Group budgets ({group_budget_method}): "
    #     f"{group_budgets} (total={sum(group_budgets)}, target={budget})"
    # )

    # ---- 组内 DPP + Top-K Soft Fusion ----
    all_fused_feats = []
    all_fused_pos = []
    all_kept_global_indices = []

    # 保存 Stage 2 信号供 LLM 剪枝复用
    s2_norms_parts = []
    s2_dyn_parts = []

    for g_idx, group in enumerate(groups):
        K_group = group_budgets[g_idx]

        # Step 0: 收集组内所有帧的 token
        pool_feats = []        # list of (N, D)
        pool_pos = []          # list of (3, N)
        pool_imp = []          # list of (N,)
        pool_norms = []        # list of (N,)
        pool_dyn = []          # list of (N,)
        group_passthrough = {} # local_idx -> (feat, pos)

        frames_needing_compression = []
        for local_idx, t in enumerate(group):
            frame_data = frame_features_list[t].float()
            frame_pos = frame_positions_list[t]

            pool_feats.append(frame_data)
            pool_pos.append(frame_pos)
            pool_imp.append(token_imp_per_frame[t])
            pool_norms.append(token_norms_per_frame[t])
            pool_dyn.append(token_dynamism_per_frame[t])
            frames_needing_compression.append(local_idx)

        if not pool_feats:
            continue

        # Step 1: 拼接组内 token 池
        pool_feats_cat = torch.cat(pool_feats, dim=0)    # (P, D)
        pool_pos_cat = torch.cat(pool_pos, dim=1)        # (3, P)
        pool_cls_cat = torch.cat(pool_imp, dim=0)     # (P,) 帧内归一化的 cls_attn
        pool_norms_cat = torch.cat(pool_norms, dim=0)    # (P,)
        pool_dyn_cat = torch.cat(pool_dyn, dim=0)        # (P,)

        # ---- 组内归一化: dynamism ----
        # cls_attn (pool_cls_cat / pool_norms_cat) 已帧内归一化, 保持不变
        # dynamism: 跨帧绝对值可比 → 组内统一归一化
        pool_dyn_cat = _td_normalize(pool_dyn_cat)
        # pool_cls_cat = _td_normalize(pool_cls_cat)

        # ---- 组内 importance = convex combination ----
        pool_imp_cat = w_cls * pool_norms_cat + w_dyn * pool_dyn_cat
        # _td_logger.info(
        #     f"[Compress] Group {g_idx} importance (组内归一化): imp=({pool_imp_cat.min().item():.3f},{pool_imp_cat.max().item():.3f}) cls=({pool_norms_cat.min().item():.3f},{pool_norms_cat.max().item():.3f}) dyn=({pool_dyn_cat.min().item():.3f},{pool_dyn_cat.max().item():.3f})"
        # )
        if pool_imp_cat.sum() == 0:
            pool_imp_cat = torch.ones_like(pool_imp_cat)

        # [批量化] 直接在 GPU 上构造 frame_ids / frame_global, 避免 list.extend + host→device
        G = len(group)
        group_t = torch.tensor(group, device=device, dtype=torch.long)              # (G,)
        pool_frame_ids_t = torch.arange(G, device=device, dtype=torch.long)\
                                .repeat_interleave(N)                                # (G*N,)
        pool_frame_global_t = group_t.repeat_interleave(N)                           # (G*N,)

        P = pool_feats_cat.shape[0]
        K_group = min(K_group, P)  # 不能超过池大小

        # Build pool-to-global mapping
        # [批量化] (group_t * N).unsqueeze(1) + arange(N) → flatten,逐元素等价于双层 for
        pool_to_global = (
            group_t.unsqueeze(1) * N + torch.arange(N, device=device, dtype=torch.long)
        ).flatten()                                                                  # (G*N,)

        if K_group >= P:
            # 不需要压缩: 所有 token 保留
            all_fused_feats.append(pool_feats_cat.to(dtype))
            all_fused_pos.append(pool_pos_cat)
            all_kept_global_indices.append(pool_to_global)  # all tokens kept
            s2_norms_parts.append(pool_norms_cat)
            s2_dyn_parts.append(pool_dyn_cat)
            continue

        # ---- Pre-split 策略 (v8) ----
        # 预先拆分: DPP 拿 K_anchor, trash pool 拿 K_residual, 总和 = K_group
        _want_residual = (residual_ratio > 0 and K_group > 1)
        if _want_residual:
            K_residual = max(1, int(K_group * residual_ratio))
            K_anchor = K_group - K_residual
        else:
            K_residual = 0
            K_anchor = K_group

        # Step 2: 每帧保底 min_per_frame 个 anchor (时间覆盖保障)
        # [批量化] 等长帧 + 池内按帧顺序排列 → 直接 view(G, N).topk(k, dim=1)
        # 前提: P == G * N, 池按 group[0] 全部 token, group[1] 全部 token, ... 顺序拼接
        # min_tokens_per_frame 可能 > 1 也支持 (这里硬编码=1, 但保留通用逻辑)
        _n_force = min(min_tokens_per_frame, N)
        if _n_force > 0 and P == G * N:
            imp_per_frame = pool_imp_cat.view(G, N)                          # (G, N)
            _, idx_in_frame = imp_per_frame.topk(_n_force, dim=1)            # (G, k)
            offsets = torch.arange(G, device=device, dtype=torch.long).unsqueeze(1) * N
            init_selected = (idx_in_frame + offsets).flatten()               # (G*k,)
        else:
            init_selected = None

        # 确保强制选中的数量不超过 K_group
        if init_selected is not None and init_selected.shape[0] >= K_anchor:
            anchor_indices = init_selected[:K_anchor]
        elif anchor_method == "dpc_knn":
            # DPC-kNN anchor 选择: 全向量化, 无 Python 循环, 比 DPP 快 5-20x
            _, anchor_indices = _dpc_knn_cluster(
                features=pool_feats_cat,
                num_clusters=K_anchor,
                k=min(7, P - 1),
                importance=pool_imp_cat,
            )
            # 合并 init_selected (保底帧 anchor)
            if init_selected is not None and init_selected.numel() > 0:
                # 去重: init_selected 中不在 anchor_indices 里的补入
                _existing = set(anchor_indices.tolist())
                _extra = [idx for idx in init_selected.tolist() if idx not in _existing]
                if _extra:
                    # 替换 anchor_indices 中 importance 最低的
                    _anc_imp = pool_imp_cat[anchor_indices]
                    n_replace = min(len(_extra), K_anchor)
                    _, _weakest = _anc_imp.topk(n_replace, largest=False)
                    for _r_idx, _e_idx in enumerate(_extra[:n_replace]):
                        anchor_indices[_weakest[_r_idx]] = _e_idx
        elif anchor_method == "leverage":
            # 帧保底 + importance 融合 + 扩大 q
            q = min(4 * K_anchor, P, pool_feats_cat.shape[1])
            U, S, V = torch.svd_lowrank(pool_feats_cat.float(), q=q)
            leverage = (U ** 2).sum(dim=1)
            score = leverage * pool_imp_cat           # 融合 importance
            # score = leverage          # 不融合 importance
            score[init_selected] = -1                 # 保底的排除
            _, extra = score.topk(K_anchor - len(init_selected))
            anchor_indices = torch.cat([init_selected, extra])
        
        elif anchor_method == "facility_location":
            anchor_indices = _facility_location_select(
                features=pool_feats_cat,
                importance=pool_imp_cat,
                k=K_anchor,
                init_selected=init_selected,
            )
        elif anchor_method == "facility_location_optimized":
            anchor_indices = _facility_location_select_optimized(
                features=pool_feats_cat,
                importance=pool_imp_cat,
                k=K_anchor,
                init_selected=init_selected,
            )
        elif anchor_method == "dpp_optimized":
            anchor_indices = _dpp_select_optimized(pool_feats_cat, pool_imp_cat, K_anchor,
                                         init_selected=init_selected)
        else:
            # Step 3: 组内 DPP anchor 选择 (从 init_selected 开始)
            anchor_indices = _dpp_select(pool_feats_cat, pool_imp_cat, K_anchor,
                                         init_selected=init_selected)

        # Step 4: 构建 anchor/drop 分割, 调用 Top-K 软融合
        anchor_mask = torch.zeros(P, dtype=torch.bool, device=device)
        anchor_mask[anchor_indices] = True

        anchor_feats = pool_feats_cat[anchor_indices]      # (K_anchor, D)
        anchor_pos = pool_pos_cat[:, anchor_indices]        # (3, K_anchor)
        anchor_frame_id = pool_frame_ids_t[anchor_indices]  # (K_anchor,) local frame id

        # Drop tokens: 非 anchor → Top-K 软分配 + scatter softmax 融合
        drop_mask = ~anchor_mask
        _need_residual = _want_residual  # 需要 residual_info 来判断 trash pool
        residual_info = None

        if drop_mask.any():
            _fuse_result = _topk_trash_fuse(
                all_feats=pool_feats_cat,
                all_pos=pool_pos_cat,
                all_imp=pool_imp_cat,
                anchor_indices=anchor_indices,
                all_frame_ids=pool_frame_ids_t,
                top_k=topk_fusion,
                trash_ratio=trash_ratio,
                cross_frame_lambda=cross_frame_lambda,
                fusion_temperature=fusion_temperature,
                fusion_method=fusion_method,
                anchor_weight=anchor_weight,
                return_residual_info=_need_residual,
            )
            if _need_residual:
                fused_feat, fused_pos, residual_info = _fuse_result
            else:
                fused_feat, fused_pos = _fuse_result
        else:
            fused_feat = anchor_feats
            fused_pos = anchor_pos

        fused_feat = fused_feat.to(dtype)

        # ---- Pre-split: trash pool 聚类补充 (v8) ----
        # DPP 选出的 K_anchor 个 anchor 已完成融合, 现在用 trash pool 填充剩余 K_residual 预算
        _res_pool_indices_in_pool = None
        if _want_residual and residual_info is not None and residual_info['count'] > 0:
            _res_tw = residual_info['trash_weights']        # (N_drop,)
            _res_df = residual_info['drop_feats']           # (N_drop, D)
            _res_dp = residual_info['drop_pos']             # (3, N_drop)
            _res_di = residual_info['drop_imp']             # (N_drop,)
            _res_dfid = residual_info['drop_frame_ids']     # (N_drop,) local frame id
            _res_dmi = residual_info['drop_mask_indices']   # (N_drop,) pool_feats_cat indices

            # 过滤 trash_weight 接近 0 的 drop (信息已被 anchor 充分吸收)
            _valid_res = (_res_tw > 0.01)
            # _valid_res = (_res_tw > (trash_ratio / topk_fusion))
            _n_valid = int(_valid_res.sum().item())

            if _n_valid > 0 and _n_valid >= K_residual * min_compression_ratio:
                # trash pool 足够大, 值得做聚类补充
                _vr_feats = _res_tw[_valid_res].unsqueeze(-1) * _res_df[_valid_res]
                _vr_imp = _res_tw[_valid_res] * _res_di[_valid_res]
                if anchor_weight >= 1.0 and res_imp_alpha < 1.0:
                    _vr_imp = _vr_imp ** res_imp_alpha
                _vr_pos = _res_dp[:, _valid_res]
                _vr_fid = _res_dfid[_valid_res]
                _vr_pool_idx = _res_dmi[_valid_res]

                K_replace = min(K_residual, _n_valid)  # 不超过 trash pool 大小

                if res_method == 'dpc_knn':
                    # dpc_knn 聚类
                    # 不带 importance 跑一次
                    # feats_old, idx_old = _dpc_knn_cluster(
                    #     features=_vr_feats, num_clusters=K_replace,
                    #     k=min(7, _n_valid - 1), importance=None,
                    # )

                    # 带 importance 跑一次
                    feats_old, idx_old = _dpc_knn_cluster(
                        features=_vr_feats, num_clusters=K_replace,
                        k=min(7, _n_valid - 1), importance=_vr_imp,
                    )
                    replace_feats, _replace_center_idx = feats_old, idx_old


                    # # kmeans聚类
                    # replace_feats, _replace_center_idx = _kmeans_cluster(
                    #         features=_vr_feats,
                    #         num_clusters=K_replace,
                    #         importance=None,
                    #     )

                    # replace_feats, _replace_center_idx = _dpc_knn_cluster(
                    #     features=_vr_feats,
                    #     num_clusters=K_replace,
                    #     k=min(7, _n_valid - 1),
                    # )


                    replace_pos = _vr_pos[:, _replace_center_idx]
                    _replace_pool_idx = _vr_pool_idx[_replace_center_idx]

                else:
                    _replace_anc_idx = _dpp_select(_vr_feats, _vr_imp, K_replace)
                    replace_feats, replace_pos = _topk_trash_fuse(
                        all_feats=_vr_feats,
                        all_pos=_vr_pos,
                        all_imp=_vr_imp,
                        anchor_indices=_replace_anc_idx,
                        all_frame_ids=_vr_fid,
                        top_k=topk_fusion,
                        trash_ratio=0,
                        cross_frame_lambda=cross_frame_lambda,
                        fusion_temperature=fusion_temperature,
                        fusion_method=fusion_method,
                        anchor_weight=anchor_weight,
                    )
                    _replace_pool_idx = _vr_pool_idx[_replace_anc_idx]

                # 直接 append 到 fused_feat / fused_pos 末尾 (不替换任何 anchor)
                fused_feat = torch.cat([fused_feat, replace_feats.to(dtype)], dim=0)
                fused_pos = torch.cat([fused_pos, replace_pos], dim=1)

                # 更新 anchor_indices: 追加 trash pool 中的索引
                anchor_indices = torch.cat([anchor_indices, _replace_pool_idx])
                _res_pool_indices_in_pool = _replace_pool_idx  # 记录补充来源

                # _td_logger.info(
                #     f"[Compress] Group {g_idx}: pre-split append "
                #     f"K_group={K_group}, K_anchor={K_anchor}, K_residual={K_replace}, "
                #     f"trash_pool={_n_valid}/{residual_info['count']}, "
                #     f"min_ratio={min_compression_ratio}, "
                #     f"mean_trash_w={_res_tw[_valid_res].mean().item():.3f}"
                # )
            else:
                # trash pool 不够大: 把 K_residual 预算还给 DPP (重选 K_group 个 anchor)
                if _n_valid > 0:
                    _td_logger.info(
                        f"[Compress] Group {g_idx}: trash pool too small for append "
                        f"(trash={_n_valid}, need>={K_residual * min_compression_ratio}), "
                        f"fallback to full K_group DPP"
                    )
                else:
                    _td_logger.info(
                        f"[Compress] Group {g_idx}: trash pool empty, fallback to full K_group DPP"
                    )
                # Fallback: 用 K_group 重新选 anchor 并融合
                K_anchor_fb = K_group
                if init_selected is not None and init_selected.shape[0] >= K_anchor_fb:
                    anchor_indices_fb = init_selected[:K_anchor_fb]
                elif anchor_method == "dpc_knn":
                    _, anchor_indices_fb = _dpc_knn_cluster(
                        features=pool_feats_cat, num_clusters=K_anchor_fb,
                        k=min(7, P - 1), importance=pool_imp_cat,
                    )
                    if init_selected is not None and init_selected.numel() > 0:
                        _existing = set(anchor_indices_fb.tolist())
                        _extra = [idx for idx in init_selected.tolist() if idx not in _existing]
                        if _extra:
                            _anc_imp = pool_imp_cat[anchor_indices_fb]
                            n_replace_fb = min(len(_extra), K_anchor_fb)
                            _, _weakest = _anc_imp.topk(n_replace_fb, largest=False)
                            for _r_idx, _e_idx in enumerate(_extra[:n_replace_fb]):
                                anchor_indices_fb[_weakest[_r_idx]] = _e_idx
                elif anchor_method == "leverage":
                    q = min(4 * K_anchor_fb, P, pool_feats_cat.shape[1])
                    U, S, V = torch.svd_lowrank(pool_feats_cat.float(), q=q)
                    leverage = (U ** 2).sum(dim=1)
                    score = leverage * pool_imp_cat
                    if init_selected is not None:
                        score[init_selected] = -1
                    _, extra = score.topk(K_anchor_fb - (len(init_selected) if init_selected is not None else 0))
                    anchor_indices_fb = torch.cat([init_selected, extra]) if init_selected is not None else extra
                elif anchor_method == "facility_location":
                    anchor_indices_fb = _facility_location_select(
                        features=pool_feats_cat, importance=pool_imp_cat,
                        k=K_anchor_fb, init_selected=init_selected,
                    )
                elif anchor_method == "facility_location_optimized":
                    anchor_indices_fb = _facility_location_select_optimized(
                        features=pool_feats_cat, importance=pool_imp_cat,
                        k=K_anchor_fb, init_selected=init_selected,
                    )
                elif anchor_method == "dpp_optimized":
                    anchor_indices_fb = _dpp_select_optimized(pool_feats_cat, pool_imp_cat, K_anchor_fb,
                                                 init_selected=init_selected)
                else:
                    anchor_indices_fb = _dpp_select(pool_feats_cat, pool_imp_cat, K_anchor_fb,
                                                 init_selected=init_selected)

                anchor_mask_fb = torch.zeros(P, dtype=torch.bool, device=device)
                anchor_mask_fb[anchor_indices_fb] = True
                drop_mask_fb = ~anchor_mask_fb

                if drop_mask_fb.any():
                    fused_feat, fused_pos = _topk_trash_fuse(
                        all_feats=pool_feats_cat,
                        all_pos=pool_pos_cat,
                        all_imp=pool_imp_cat,
                        anchor_indices=anchor_indices_fb,
                        all_frame_ids=pool_frame_ids_t,
                        top_k=topk_fusion,
                        trash_ratio=trash_ratio,
                        cross_frame_lambda=cross_frame_lambda,
                        fusion_temperature=fusion_temperature,
                        fusion_method=fusion_method,
                        anchor_weight=anchor_weight,
                        return_residual_info=False,
                    )
                else:
                    fused_feat = pool_feats_cat[anchor_indices_fb]
                    fused_pos = pool_pos_cat[:, anchor_indices_fb]
                fused_feat = fused_feat.to(dtype)
                anchor_indices = anchor_indices_fb

        # 保存 s2 信号 (anchor + trash pool 补充)
        _s2_anchor_idx = anchor_indices  # v8: anchor_indices 含 DPP anchor + trash append
        s2_norms_parts.append(pool_norms_cat[_s2_anchor_idx])
        s2_dyn_parts.append(pool_dyn_cat[_s2_anchor_idx])

        # Global indices of kept tokens
        kept_global = pool_to_global[_s2_anchor_idx]

        # Step 5: 按帧拆分输出
        anchor_global_frame = pool_frame_global_t[_s2_anchor_idx]

        for local_idx, t in enumerate(group):
            # 该帧的 anchor indices (在 fused_feat 中的位置)
            _frame_anchor_mask = (anchor_global_frame == t)
            _frame_anchor_indices = _frame_anchor_mask.nonzero(as_tuple=True)[0]
            if _frame_anchor_indices.shape[0] > 0:
                all_fused_feats.append(fused_feat[_frame_anchor_indices])
                all_fused_pos.append(fused_pos[:, _frame_anchor_indices])
                all_kept_global_indices.append(kept_global[_frame_anchor_indices])
            else:
                # 该帧没有 anchor (不应该发生, 因为有 min_per_frame 保障)
                # fallback: 取该帧 importance 最高的 1 个 token
                # _td_logger.warning(f"[Compress] Frame {t} has no anchor in group DPP, using fallback")
                _fb_data = frame_features_list[t].float()
                _fb_imp = token_imp_per_frame[t]
                _fb_idx = _fb_imp.argmax().unsqueeze(0)
                all_fused_feats.append(_fb_data[_fb_idx].to(dtype))
                all_fused_pos.append(frame_positions_list[t][:, _fb_idx])
                all_kept_global_indices.append(torch.tensor([t * N + _fb_idx.item()], device=device, dtype=torch.long))

    compressed_embeds = torch.cat(all_fused_feats, dim=0)
    compressed_positions = torch.cat(all_fused_pos, dim=1)
    kept_global_indices = torch.cat(all_kept_global_indices)

    # 保存 Stage 2 信号供 LLM 剪枝复用
    config._s2_precomputed_token_norms = torch.cat(s2_norms_parts) if s2_norms_parts else None
    config._s2_precomputed_token_dyn = torch.cat(s2_dyn_parts) if s2_dyn_parts else None

    config._s2_precomputed_cls_attn = config._s2_precomputed_token_norms  # 复用 L2 norm 作为 CLS proxy

    actual_total = compressed_embeds.shape[0]
    _total_real = sum(_N_real)
    _td_logger.info(
        f"[Compress] Done: {_total_real} → {actual_total} tokens "
        f"(budget={budget}, groups={len(groups)}, method=DPP+TopK)"
    )

    return compressed_embeds, compressed_positions.to(frame_positions_list[0].dtype), kept_global_indices

# ============================================================================
# LLM 内部单层 attention-score-only 剪枝
# ============================================================================


def query_guided_pruning_qwen(
    hidden_states: torch.Tensor,       # (B, S, D)
    visual_token_range: tuple,         # (start, end)
    target_budget: int,                # 最终保留的视觉token数
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    text_position_ids: Optional[torch.Tensor] = None,
    cache_position: Optional[torch.Tensor] = None,
    past_key_values=None,
    td_config=None,
    decoder_layer=None,                # 剪枝层的 decoder_layer (用于获取 q_proj, k_proj)
    position_embeddings=None,          # RoPE embeddings (cos, sin) 用于 Q/K 旋转
    attn_history=None,                 # list of (B, N_vis) tensors from trend observe layers
    attn_scores: Optional[torch.Tensor] = None,  # (B, V) 由 decoder layer output_attentions 产出
) -> tuple:
    """LLM Decoder 内部单层 attention-score-based 剪枝。

    使用该层实际的 Q/K 投影计算注意力分数:
    1. 通过 decoder_layer 的 q_proj/k_proj 将 hidden_states 投影到 Q/K 空间
    2. 应用 RoPE 旋转位置编码
    3. 根据 llm_prune_method 选择评分范围:
       - "text_token": Q 来自 text tokens, K 来自 visual tokens
       - "last_token": Q 仅来自最后一个 token
       - "all_token": Q 来自所有 tokens, K 来自 visual tokens
    4. 对 head 维度取均值得到最终 per-visual-token 分数
    5. 融合 attn_history (Trend Observe 多层观测均值, 权重由 td_config.trend_weight 控制)
    6. Top-k 选择, 物理移除 token

    当 attn_scores 不为 None 时, 直接使用 (output_attentions 方案);
    否则走原来的手动 q_proj/k_proj 计算逻辑 (向后兼容)。

    Args:
        hidden_states: (B, S, D)
        visual_token_range: (start, end)
        target_budget: 剪枝后保留的视觉token数
        attention_mask, position_ids, text_position_ids, cache_position: 序列张量
        past_key_values: KV cache
        td_config: TensorDecompConfig
        decoder_layer: 剪枝层的 decoder_layer (含 self_attn.q_proj, k_proj)
        position_embeddings: (cos, sin) RoPE embeddings
        attn_history: list of (B, N_vis) tensors from trend observe layers
        attn_scores: (B, V) 由 decoder layer output_attentions 产出

    Returns:
        (hidden_states, attention_mask, position_ids, text_position_ids,
         cache_position, num_pruned, keep_idx)
    """
    B, S, D = hidden_states.shape
    assert B == 1, "query_guided_pruning_qwen currently supports batch_size=1 only"
    v_start, v_end = visual_token_range
    num_visual = v_end - v_start

    # 安全检查
    if v_end > S or v_start >= S or num_visual <= 0:
        return hidden_states, attention_mask, position_ids, text_position_ids, cache_position, 0, None

    if num_visual <= target_budget:
        return hidden_states, attention_mask, position_ids, text_position_ids, cache_position, 0, None

    # 确定评分方法
    prune_method = getattr(td_config, 'llm_prune_method', 'text_token') if td_config else 'text_token'

    # ========================================================================
    # 优先使用外部传入的 attn_scores (output_attentions 方案)
    # ========================================================================
    if attn_scores is not None:
        scores = attn_scores  # (B, V) — 已由 decoder layer 正确计算
        # _td_logger.info(
        #     f"[QueryPrune] Using decoder-layer attn_scores: "
        #     f"[{scores.min().item():.6f}, {scores.max().item():.6f}]"
        # )
    elif decoder_layer is not None and position_embeddings is not None:
        self_attn = decoder_layer.self_attn
        head_dim = self_attn.head_dim
        num_heads = self_attn.config.num_attention_heads
        num_kv_heads = self_attn.config.num_key_value_heads
        num_kv_groups = num_heads // num_kv_heads

        with torch.no_grad():
            # -- 计算 Q 和 K 投影 --
            # Q: 需要来自 query tokens (text 或 all)
            # K: 需要来自 visual tokens
            if prune_method == "text_token":
                # Q 来自 text tokens
                text_prefix_h = hidden_states[:, :v_start, :]       # (B, text_prefix_len, D)
                text_suffix_h = hidden_states[:, v_end:, :]         # (B, text_suffix_len, D)
                if text_prefix_h.shape[1] + text_suffix_h.shape[1] == 0:
                    return hidden_states, attention_mask, position_ids, text_position_ids, cache_position, 0, None
                query_hidden = torch.cat([text_prefix_h, text_suffix_h], dim=1)  # (B, T, D)
            elif prune_method == "last_token":
                # Q 仅来自最后一个 token
                query_hidden = hidden_states[:, -1:, :]  # (B, 1, D)
            else:
                # Q 来自所有 tokens
                query_hidden = hidden_states  # (B, S, D)

            visual_hidden = hidden_states[:, v_start:v_end, :]  # (B, V, D)

            # 经过 q_proj, k_proj 线性投影
            q_states = self_attn.q_proj(query_hidden)   # (B, T_or_S, num_heads * head_dim)
            k_states = self_attn.k_proj(visual_hidden)  # (B, V, num_kv_heads * head_dim)

            # reshape 为 multi-head 格式
            q_len = q_states.shape[1]
            v_len = k_states.shape[1]
            q_states = q_states.view(B, q_len, num_heads, head_dim).transpose(1, 2)    # (B, H, T, d)
            k_states = k_states.view(B, v_len, num_kv_heads, head_dim).transpose(1, 2) # (B, Hkv, V, d)

            # -- 应用 RoPE --
            cos, sin = position_embeddings
            # cos/sin shape: (B or 1, S, head_dim) — 需要切片到对应位置
            if cos.ndim == 3:
                # cos: (B, S, d) — 取出 query 和 visual 对应的位置
                if prune_method == "text_token":
                    # query 位置: text_prefix + text_suffix
                    q_cos = torch.cat([cos[:, :v_start, :], cos[:, v_end:, :]], dim=1)
                    q_sin = torch.cat([sin[:, :v_start, :], sin[:, v_end:, :]], dim=1)
                elif prune_method == "last_token":
                    q_cos = cos[:, -1:, :]
                    q_sin = sin[:, -1:, :]
                else:
                    q_cos = cos
                    q_sin = sin
                k_cos = cos[:, v_start:v_end, :]
                k_sin = sin[:, v_start:v_end, :]
            elif cos.ndim == 2:
                # cos: (S, d)
                if prune_method == "text_token":
                    q_cos = torch.cat([cos[:v_start, :], cos[v_end:, :]], dim=0).unsqueeze(0)
                    q_sin = torch.cat([sin[:v_start, :], sin[v_end:, :]], dim=0).unsqueeze(0)
                elif prune_method == "last_token":
                    q_cos = cos[-1:, :].unsqueeze(0)
                    q_sin = sin[-1:, :].unsqueeze(0)
                else:
                    q_cos = cos.unsqueeze(0)
                    q_sin = sin.unsqueeze(0)
                k_cos = cos[v_start:v_end, :].unsqueeze(0)
                k_sin = sin[v_start:v_end, :].unsqueeze(0)
            else:
                # 无法确定 cos/sin 形状, fallback
                q_cos = k_cos = cos
                q_sin = k_sin = sin

            # -- 计算注意力 logits --
            # GQA: repeat K to match Q heads
            if num_kv_groups > 1:
                k_states = repeat_kv(k_states, num_kv_groups)  # (B, H, V, d)

            # attn_logits: (B, H, T, V)
            scale = 1.0 / math.sqrt(head_dim)
            attn_logits = torch.matmul(q_states, k_states.transpose(-2, -1)) * scale

            # -- 在 visual 维度做 softmax (每个 query token 独立归一化) --
            attn_weights = attn_logits.softmax(dim=-1)  # (B, H, T, V)

            # -- 聚合: max over query tokens, mean over heads → (B, V) --
            if prune_method == "last_token":
                # 只有 1 个 query token, 直接 squeeze
                scores = attn_weights[:, :, 0, :].mean(dim=1)  # (B, V)
            else:
                scores = attn_weights.max(dim=2).values.mean(dim=1)  # (B, V)

        # _td_logger.info(
        #     f"[QueryPrune] method={prune_method}, "
        #     f"Q_shape=({q_len},), K_shape=({v_len},), "
        #     f"scores: [{scores.min().item():.6f}, {scores.max().item():.6f}]"
        # )

    else:
        # ========================================================================
        # Fallback: 使用 cosine similarity (当 decoder_layer 不可用时)
        # ========================================================================
        _td_logger.warning(
            "[QueryPrune] decoder_layer or position_embeddings not available, "
            "falling back to cosine similarity"
        )
        visual_hidden = hidden_states[:, v_start:v_end, :]
        text_prefix = hidden_states[:, :v_start, :]
        text_suffix = hidden_states[:, v_end:, :]

        if text_prefix.shape[1] + text_suffix.shape[1] == 0:
            return hidden_states, attention_mask, position_ids, text_position_ids, cache_position, 0, None

        text_all = torch.cat([text_prefix, text_suffix], dim=1)
        query = text_all.mean(dim=1, keepdim=True)

        query_normed = F.normalize(query, dim=-1)
        visual_normed = F.normalize(visual_hidden, dim=-1)
        scores = (visual_normed * query_normed).sum(dim=-1)  # (B, num_visual)

        # _td_logger.info(
        #     f"[QueryPrune] fallback cosine, scores: [{scores.min().item():.3f}, {scores.max().item():.3f}]"
        # )

    # ========================================================================
    # 融合 attn_history (Trend Observe 多层观测分数)
    # ========================================================================
    if attn_history is not None and len(attn_history) > 0:
        # attn_history: list of (B, N_vis) tensors
        # 取均值作为趋势信号, 与当前层 scores 加权融合
        _trend_weight = getattr(td_config, 'trend_weight', 0.3) if td_config else 0.3
        _history_stack = torch.stack(attn_history, dim=0)  # (L, B, N_vis)
        _history_mean = _history_stack.mean(dim=0)  # (B, N_vis)
        # 归一化到 [0,1]
        _h_min = _history_mean.min(dim=-1, keepdim=True).values
        _h_max = _history_mean.max(dim=-1, keepdim=True).values
        _h_range = (_h_max - _h_min).clamp(min=1e-8)
        _history_norm = (_history_mean - _h_min) / _h_range

        _s_min = scores.min(dim=-1, keepdim=True).values
        _s_max = scores.max(dim=-1, keepdim=True).values
        _s_range = (_s_max - _s_min).clamp(min=1e-8)
        _scores_norm = (scores - _s_min) / _s_range

        scores = (1 - _trend_weight) * _scores_norm + _trend_weight * _history_norm
        # _td_logger.info(
        #     f"[QueryPrune] Fused {len(attn_history)} trend layers (weight={_trend_weight}), "
        #     f"final scores: [{scores.min().item():.6f}, {scores.max().item():.6f}]"
        # )

    # ========================================================================
    # Top-k selection & 物理移除
    # ========================================================================
    visual_hidden = hidden_states[:, v_start:v_end, :]
    k = min(target_budget, num_visual)
    _, top_indices = scores.topk(k, dim=1)
    top_indices = top_indices.sort(dim=1).values

    keep_idx = top_indices[0]  # (k,)
    kept_visual = visual_hidden[0, keep_idx, :].unsqueeze(0)

    # 重新拼接
    text_prefix = hidden_states[:, :v_start, :]
    text_suffix = hidden_states[:, v_end:, :]
    new_hidden = torch.cat([text_prefix, kept_visual, text_suffix], dim=1)

    # 更新 attention_mask
    new_attention_mask = attention_mask
    if attention_mask is not None and attention_mask.ndim == 2:
        prefix_mask = attention_mask[:, :v_start]
        suffix_mask = attention_mask[:, v_end:]
        visual_mask = torch.ones(B, k, device=attention_mask.device, dtype=attention_mask.dtype)
        new_attention_mask = torch.cat([prefix_mask, visual_mask, suffix_mask], dim=1)

    # 更新 3D RoPE position_ids
    new_position_ids = position_ids
    if position_ids is not None:
        prefix_pos = position_ids[:, :, :v_start]
        suffix_pos = position_ids[:, :, v_end:]
        visual_pos = position_ids[:, :, v_start:v_end]
        kept_pos = visual_pos[:, :, keep_idx]
        new_position_ids = torch.cat([prefix_pos, kept_pos, suffix_pos], dim=2)

    # 更新 2D text_position_ids
    new_text_position_ids = text_position_ids
    if text_position_ids is not None and text_position_ids.ndim == 2:
        tp_prefix = text_position_ids[:, :v_start]
        tp_suffix = text_position_ids[:, v_end:]
        tp_visual = text_position_ids[:, v_start:v_end]
        tp_kept = tp_visual[:, keep_idx]
        new_text_position_ids = torch.cat([tp_prefix, tp_kept, tp_suffix], dim=1)

    # 更新 cache_position
    new_cache_position = cache_position
    if cache_position is not None:
        new_cache_position = torch.arange(0, new_hidden.shape[1], device=cache_position.device)

    # 清理 past_key_values
    if past_key_values is not None and hasattr(past_key_values, 'key_cache'):
        _kv_keep = torch.cat([
            torch.arange(v_start, device=keep_idx.device),
            v_start + keep_idx,
            torch.arange(v_end, S, device=keep_idx.device),
        ])
        for _kv_layer_idx in range(len(past_key_values.key_cache)):
            if past_key_values.key_cache[_kv_layer_idx].numel() > 0:
                kv_len = past_key_values.key_cache[_kv_layer_idx].shape[2]
                if kv_len == S:
                    past_key_values.key_cache[_kv_layer_idx] = past_key_values.key_cache[_kv_layer_idx].index_select(2, _kv_keep)
                    past_key_values.value_cache[_kv_layer_idx] = past_key_values.value_cache[_kv_layer_idx].index_select(2, _kv_keep)

    num_pruned = num_visual - k
    _td_logger.info(
        f"[QueryPrune] {num_visual} → {k} visual tokens (pruned {num_pruned})"
    )

    return new_hidden, new_attention_mask, new_position_ids, new_text_position_ids, new_cache_position, num_pruned, keep_idx


# ============================================================================
# Qwen3-VL 专用: Attention-score-based query-guided 剪枝 (方案A)
# ============================================================================

def query_guided_pruning_qwen3(
    hidden_states: torch.Tensor,       # (B, S, D)
    visual_token_range: tuple,         # (start, end)
    target_budget: int,                # 最终保留的视觉token数
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    text_position_ids: Optional[torch.Tensor] = None,
    cache_position: Optional[torch.Tensor] = None,
    past_key_values=None,
    td_config=None,
    position_embeddings=None,          # RoPE embeddings (cos, sin)
    attn_scores: Optional[torch.Tensor] = None,   # (B, V) 由上一层 decoder 正常 forward 产出
    attn_history=None,                 # list of (B, N_vis) tensors from trend observe layers
) -> tuple:
    """LLM Decoder 内部 attention-score-based 剪枝 (Qwen3 专用, 方案A)。

    与 query_guided_pruning_qwen 的区别:
      - 不自己计算 Q/K attention (避免遗漏 q_norm/k_norm/RoPE)
      - 直接接收由 decoder layer 正常 forward 产出的 attn_scores
      - attn_scores 已经是正确的 (经过 q_norm + k_norm + RoPE + softmax)

    Args:
        hidden_states: (B, S, D)
        visual_token_range: (start, end)
        target_budget: 剪枝后保留的视觉token数
        attention_mask, position_ids, text_position_ids, cache_position: 序列张量
        past_key_values: KV cache
        td_config: TensorDecompConfig
        position_embeddings: (cos, sin) RoPE embeddings, 用于剪枝后重建
        attn_scores: (B, V) 每个 visual token 的重要性分数
            由 Qwen3VLTextAttention_forward 中正确计算 (含 q_norm/k_norm/RoPE)
        attn_history: list of (B, N_vis) tensors from trend observe layers

    Returns:
        (hidden_states, attention_mask, position_ids, text_position_ids,
         cache_position, num_pruned, keep_idx)
    """
    B, S, D = hidden_states.shape
    assert B == 1, "query_guided_pruning_qwen3 currently supports batch_size=1 only"
    v_start, v_end = visual_token_range
    num_visual = v_end - v_start

    # 安全检查
    if v_end > S or v_start >= S or num_visual <= 0:
        return hidden_states, attention_mask, position_ids, text_position_ids, cache_position, 0, None

    if num_visual <= target_budget:
        return hidden_states, attention_mask, position_ids, text_position_ids, cache_position, 0, None

    # ========================================================================
    # 使用外部传入的 attn_scores
    # ========================================================================
    if attn_scores is not None:
        scores = attn_scores  # (B, V) — 已由 decoder layer 正确计算
        # _td_logger.info(
        #     f"[QueryPrune-Q3] Using decoder-layer attn_scores: "
        #     f"[{scores.min().item():.6f}, {scores.max().item():.6f}]"
        # )
    else:
        # Fallback: cosine similarity (当 attn_scores 不可用时)
        _td_logger.warning(
            "[QueryPrune-Q3] attn_scores not available, falling back to cosine similarity"
        )
        visual_hidden = hidden_states[:, v_start:v_end, :]
        text_prefix = hidden_states[:, :v_start, :]
        text_suffix = hidden_states[:, v_end:, :]

        if text_prefix.shape[1] + text_suffix.shape[1] == 0:
            return hidden_states, attention_mask, position_ids, text_position_ids, cache_position, 0, None

        text_all = torch.cat([text_prefix, text_suffix], dim=1)
        query = text_all.mean(dim=1, keepdim=True)

        query_normed = F.normalize(query, dim=-1)
        visual_normed = F.normalize(visual_hidden, dim=-1)
        scores = (visual_normed * query_normed).sum(dim=-1)  # (B, num_visual)

    # ========================================================================
    # 融合 attn_history (Trend Observe 多层观测分数)
    # ========================================================================
    if attn_history is not None and len(attn_history) > 0:
        _trend_weight = getattr(td_config, 'trend_weight', 0.3) if td_config else 0.3
        _history_stack = torch.stack(attn_history, dim=0)  # (L, B, N_vis)
        _history_mean = _history_stack.mean(dim=0)  # (B, N_vis)
        # 归一化到 [0,1]
        _h_min = _history_mean.min(dim=-1, keepdim=True).values
        _h_max = _history_mean.max(dim=-1, keepdim=True).values
        _h_range = (_h_max - _h_min).clamp(min=1e-8)
        _history_norm = (_history_mean - _h_min) / _h_range

        _s_min = scores.min(dim=-1, keepdim=True).values
        _s_max = scores.max(dim=-1, keepdim=True).values
        _s_range = (_s_max - _s_min).clamp(min=1e-8)
        _scores_norm = (scores - _s_min) / _s_range

        scores = (1 - _trend_weight) * _scores_norm + _trend_weight * _history_norm
        # _td_logger.info(
        #     f"[QueryPrune-Q3] Fused {len(attn_history)} trend layers (weight={_trend_weight}), "
        #     f"final scores: [{scores.min().item():.6f}, {scores.max().item():.6f}]"
        # )

    # ========================================================================
    # Top-k selection & 物理移除
    # ========================================================================
    visual_hidden = hidden_states[:, v_start:v_end, :]
    k = min(target_budget, num_visual)
    _, top_indices = scores.topk(k, dim=1)
    top_indices = top_indices.sort(dim=1).values

    keep_idx = top_indices[0]  # (k,)
    kept_visual = visual_hidden[0, keep_idx, :].unsqueeze(0)

    # 重新拼接
    text_prefix = hidden_states[:, :v_start, :]
    text_suffix = hidden_states[:, v_end:, :]
    new_hidden = torch.cat([text_prefix, kept_visual, text_suffix], dim=1)

    # 更新 attention_mask
    new_attention_mask = attention_mask
    if attention_mask is not None and attention_mask.ndim == 2:
        prefix_mask = attention_mask[:, :v_start]
        suffix_mask = attention_mask[:, v_end:]
        visual_mask = torch.ones(B, k, device=attention_mask.device, dtype=attention_mask.dtype)
        new_attention_mask = torch.cat([prefix_mask, visual_mask, suffix_mask], dim=1)

    # 更新 3D RoPE position_ids
    new_position_ids = position_ids
    if position_ids is not None:
        prefix_pos = position_ids[:, :, :v_start]
        suffix_pos = position_ids[:, :, v_end:]
        visual_pos = position_ids[:, :, v_start:v_end]
        kept_pos = visual_pos[:, :, keep_idx]
        new_position_ids = torch.cat([prefix_pos, kept_pos, suffix_pos], dim=2)

    # 更新 2D text_position_ids
    new_text_position_ids = text_position_ids
    if text_position_ids is not None and text_position_ids.ndim == 2:
        tp_prefix = text_position_ids[:, :v_start]
        tp_suffix = text_position_ids[:, v_end:]
        tp_visual = text_position_ids[:, v_start:v_end]
        tp_kept = tp_visual[:, keep_idx]
        new_text_position_ids = torch.cat([tp_prefix, tp_kept, tp_suffix], dim=1)

    # 更新 cache_position
    new_cache_position = cache_position
    if cache_position is not None:
        new_cache_position = torch.arange(0, new_hidden.shape[1], device=cache_position.device)

    # 清理 past_key_values
    if past_key_values is not None and hasattr(past_key_values, 'key_cache'):
        _kv_keep = torch.cat([
            torch.arange(v_start, device=keep_idx.device),
            v_start + keep_idx,
            torch.arange(v_end, S, device=keep_idx.device),
        ])
        for _kv_layer_idx in range(len(past_key_values.key_cache)):
            if past_key_values.key_cache[_kv_layer_idx].numel() > 0:
                kv_len = past_key_values.key_cache[_kv_layer_idx].shape[2]
                if kv_len == S:
                    past_key_values.key_cache[_kv_layer_idx] = past_key_values.key_cache[_kv_layer_idx].index_select(2, _kv_keep)
                    past_key_values.value_cache[_kv_layer_idx] = past_key_values.value_cache[_kv_layer_idx].index_select(2, _kv_keep)

    num_pruned = num_visual - k
    _td_logger.info(f"[QueryPrune-Q3] {num_visual} → {k} visual tokens (pruned {num_pruned})")

    return new_hidden, new_attention_mask, new_position_ids, new_text_position_ids, new_cache_position, num_pruned, keep_idx


# ============================================================================
# LLaVA OneVision: Post-projector DySeg-DPP 压缩 (1D global indices)
# ============================================================================


def tensor_decomp_compression_llava(
    video_embeds: Tensor,
    cls_attention: Tensor,
    num_frames: int,
    tokens_per_frame: int,
    config,
) -> Tuple[Tensor, Tensor]:
    """Post-projector DySeg-DPP 压缩 (LLaVA 版)。

    与 Qwen2.5-VL 版的区别:
      - 输入: (T, N, D) per-frame + (T, N) cls_attn (而非 flat + M-RoPE)
      - 输出: keep_visual_indices 全局索引 (而非 M-RoPE positions)
      - 无 valid_mask / padding 逻辑 (无 ViT fusion)
      - 无 position_ids_video (1D RoPE 压缩后自动重生成)

    Args:
        video_embeds: (T, N, D) 2D pooling 后的视频 token。
        cls_attention: (T, N) SigLIP 最后一层 per-token importance。
        num_frames: T。
        tokens_per_frame: N (169 after 2D pooling)。
        config: TensorDecompConfig 实例。

    Returns:
        compressed_embeds: (budget, D) 压缩后的视频 token。
        keep_visual_indices: (budget,) 全局索引 (frame_idx * N + token_idx)。
    """
    T = num_frames
    N = tokens_per_frame
    D = video_embeds.shape[-1]
    device = video_embeds.device
    dtype = video_embeds.dtype

    _s2_ratio = getattr(config, 'stage2_retention_ratio', 0.20) or 0.20
    budget = max(1, int(T * N * _s2_ratio))

    # 如果不需要压缩
    if budget >= T * N:
        # _td_logger.info(f"[Compress] SKIP: budget={budget} >= T*N={T*N}")
        all_indices = torch.arange(T * N, device=device, dtype=torch.long)
        return video_embeds.reshape(T * N, D), all_indices

    # ---- 拆分为逐帧 ----
    frame_features_list = [video_embeds[t] for t in range(T)]       # list of (N, D)
    frame_global_offsets = [t * N for t in range(T)]

    # ---- 读取配置参数 ----
    dynamism_window = config.dynamism_window
    w_dyn = config.w_dyn
    asym_w = config.asym_w
    fusion_temperature = config.fusion_temperature
    cross_frame_lambda = config.cross_frame_lambda
    dyseg_threshold = config.dyseg_threshold
    min_segment_num = getattr(config, 'min_segment_num', 0)
    complementary_segment = getattr(config, 'complementary_segment', True)
    topk_fusion = config.topk_fusion
    trash_ratio = config.trash_ratio
    fusion_method = getattr(config, 'fusion_method', 'mean')
    anchor_weight = getattr(config, 'anchor_weight', 0.5)
    anchor_method = getattr(config, 'anchor_method', 'dpp')
    residual_ratio = getattr(config, 'residual_ratio', 0.0)
    min_compression_ratio = getattr(config, 'min_compression_ratio', 3)
    res_imp_alpha = getattr(config, 'res_imp_alpha', 0.3)
    res_method = getattr(config, 'res_method', 'dpc_knn')
    group_budget_method = config.group_budget_method
    min_tokens_per_frame = 1

    # Normalize features for similarity
    frame_features_norm = [F.normalize(f.float(), dim=-1) for f in frame_features_list]

    # Signal 1: CLS attention from SigLIP
    # LLaVA 使用 SigLIP 的 attn_weights.mean(heads).mean(queries) 作为 per-token importance
    token_norms_per_frame = [cls_attention[t].float() for t in range(T)]

    # Signal 2: Dynamism
    # [优化] sim 矩阵复用: 每对相邻帧 (t, t+d) 只做一次 (N,N) matmul,
    #   max(dim=1) → fwd_max[t], max(dim=0) → bwd_max[t+d]
    w = dynamism_window

    # [优化] 批量化 dynamism 计算: 一次 bmm 代替 T-1 次独立 matmul
    all_frames_stacked = torch.stack(frame_features_norm)  # (T, N, D)

    fwd_max_all = torch.zeros(T, N, device=device, dtype=torch.float32)
    bwd_max_all = torch.zeros(T, N, device=device, dtype=torch.float32)

    for d in range(1, min(w, T - 1) + 1):
        curr = all_frames_stacked[:T - d]   # (T-d, N, D)
        next_ = all_frames_stacked[d:]      # (T-d, N, D)
        sim_batch = torch.bmm(curr, next_.transpose(1, 2))  # (T-d, N, N)
        fwd_max_all[:T - d] = torch.max(fwd_max_all[:T - d], sim_batch.max(dim=2).values)
        bwd_max_all[d:] = torch.max(bwd_max_all[d:], sim_batch.max(dim=1).values)

    # 向量化 dynamism 计算
    novelty_all = (1.0 - torch.max(fwd_max_all, bwd_max_all)).clamp(min=0)  # (T, N)
    asymmetry_all = torch.abs(fwd_max_all - bwd_max_all)                     # (T, N)
    token_dyn_all = (1.0 - asym_w) * novelty_all + asym_w * asymmetry_all    # (T, N)
    token_dynamism_per_frame = [token_dyn_all[t] for t in range(T)]

    # Normalize CLS attention (帧内归一化)
    token_norms_per_frame = [_td_normalize(tn) for tn in token_norms_per_frame]

    # _td_logger.info(
    #     f"[Compress] Signals (T={T}, N={N}): "
    #     f"dynamism formula: (1-{asym_w})*novelty + {asym_w}*asymmetry"
    # )

    # ---- DySeg 分组 ----
    groups = _dyseg_group_frames(frame_features_norm, dyseg_threshold, min_segment_num, complementary_segment)
    # _td_logger.info(
    #     f"[Compress] DySeg: threshold={dyseg_threshold}, "
    #     f"groups={len(groups)}, sizes={[len(g) for g in groups]}"
    # )

    # ---- 组预算分配 ----
    N_real = [N] * T
    group_budgets = _compute_group_budget(
        groups=groups,
        frame_features_norm=frame_features_norm,
        total_budget=budget,
        method=group_budget_method,
        N_real=N_real,
        device=device,
        token_importance_per_frame=token_norms_per_frame,
        effective_rank_k=int(getattr(config, 'effective_rank_k', 64)),
    )
    # _td_logger.info(
    #     f"[Compress] Group budgets ({group_budget_method}): "
    #     f"{group_budgets} (total={sum(group_budgets)}, target={budget})"
    # )

    # ---- 组内 DPP + Top-K Soft Fusion ----
    all_fused_feats = []
    all_fused_indices = []  # 全局索引
    w_cls = max(0.0, 1.0 - w_dyn)

    for g_idx, group in enumerate(groups):
        K_group = group_budgets[g_idx]

        # 收集组内所有帧的 token
        pool_feats = []
        pool_global_idx = []
        pool_cls = []
        pool_dyn = []
        pool_frame_ids = []

        for local_idx, t in enumerate(group):
            pool_feats.append(frame_features_list[t].float())
            pool_global_idx.append(
                torch.arange(N, device=device, dtype=torch.long) + frame_global_offsets[t]
            )
            pool_cls.append(token_norms_per_frame[t])
            pool_dyn.append(token_dynamism_per_frame[t])
            pool_frame_ids.extend([local_idx] * N)

        pool_feats_cat = torch.cat(pool_feats, dim=0)       # (P, D)
        pool_gidx_cat = torch.cat(pool_global_idx, dim=0)   # (P,)
        pool_cls_cat = torch.cat(pool_cls, dim=0)            # (P,)
        pool_dyn_cat = torch.cat(pool_dyn, dim=0)            # (P,)
        pool_frame_ids_t = torch.tensor(pool_frame_ids, device=device, dtype=torch.long)

        P = pool_feats_cat.shape[0]
        K_group = min(K_group, P)

        if K_group >= P:
            all_fused_feats.append(pool_feats_cat.to(dtype))
            all_fused_indices.append(pool_gidx_cat)
            continue

        # 组内归一化 dynamism
        pool_dyn_cat = _td_normalize(pool_dyn_cat)

        # 组内 importance = convex combination
        pool_imp_cat = w_cls * pool_cls_cat + w_dyn * pool_dyn_cat
        if pool_imp_cat.sum() == 0:
            pool_imp_cat = torch.ones_like(pool_imp_cat)

        # ---- Pre-split 策略 (v8) ----
        K_anchor = K_group - max(1, int(K_group * residual_ratio)) if (residual_ratio > 0 and K_group > 1) else K_group
        K_residual = K_group - K_anchor
        _want_residual = (K_residual > 0)

        # 每帧保底 min_per_frame 个 anchor
        unique_local_ids = pool_frame_ids_t.unique()
        init_selected_list = []
        for _lid in unique_local_ids:
            _frame_mask = (pool_frame_ids_t == _lid)
            _frame_indices = _frame_mask.nonzero(as_tuple=True)[0]
            _frame_imp = pool_imp_cat[_frame_indices]
            _n_force = min(min_tokens_per_frame, _frame_indices.shape[0])
            if _n_force > 0:
                _, _top_k = _frame_imp.topk(_n_force)
                init_selected_list.append(_frame_indices[_top_k])

        init_selected = torch.cat(init_selected_list) if init_selected_list else None

        if init_selected is not None and init_selected.shape[0] >= K_anchor:
            anchor_indices = init_selected[:K_anchor]
        elif anchor_method == "dpc_knn":
            # DPC-kNN anchor 选择: 全向量化, 无 Python 循环, 比 DPP 快 5-20x
            _, anchor_indices = _dpc_knn_cluster(
                features=pool_feats_cat,
                num_clusters=K_anchor,
                k=min(7, P - 1),
                importance=pool_imp_cat,
            )
            # 合并 init_selected (保底帧 anchor)
            if init_selected is not None and init_selected.numel() > 0:
                # 去重: init_selected 中不在 anchor_indices 里的补入
                _existing = set(anchor_indices.tolist())
                _extra = [idx for idx in init_selected.tolist() if idx not in _existing]
                if _extra:
                    # 替换 anchor_indices 中 importance 最低的
                    _anc_imp = pool_imp_cat[anchor_indices]
                    n_replace = min(len(_extra), K_anchor)
                    _, _weakest = _anc_imp.topk(n_replace, largest=False)
                    for _r_idx, _e_idx in enumerate(_extra[:n_replace]):
                        anchor_indices[_weakest[_r_idx]] = _e_idx
        elif anchor_method == "facility_location":
            anchor_indices = _facility_location_select(
                features=pool_feats_cat,
                importance=pool_imp_cat,
                k=K_anchor,
                init_selected=init_selected,
            )
        elif anchor_method == "facility_location_optimized":
            anchor_indices = _facility_location_select_optimized(
                features=pool_feats_cat,
                importance=pool_imp_cat,
                k=K_anchor,
                init_selected=init_selected,
            )
        elif anchor_method == "leverage":
            q = min(4 * K_anchor, P, pool_feats_cat.shape[1])
            U, S_val, V = torch.svd_lowrank(pool_feats_cat.float(), q=q)
            leverage = (U ** 2).sum(dim=1)
            score = leverage * pool_imp_cat
            if init_selected is not None:
                score[init_selected] = -1
            _, extra = score.topk(K_anchor - (len(init_selected) if init_selected is not None else 0))
            anchor_indices = torch.cat([init_selected, extra]) if init_selected is not None else extra
        elif anchor_method == "dpp_optimized":
            anchor_indices = _dpp_select_optimized(pool_feats_cat, pool_imp_cat, K_anchor,
                                         init_selected=init_selected)

        else:
            anchor_indices = _dpp_select(pool_feats_cat, pool_imp_cat, K_anchor,
                                         init_selected=init_selected)

        # Top-K 软融合 (使用统一 _topk_trash_fuse, 传入 1D positions)
        anchor_mask = torch.zeros(P, dtype=torch.bool, device=device)
        anchor_mask[anchor_indices] = True
        drop_mask = ~anchor_mask
        _need_residual = _want_residual
        residual_info = None

        if drop_mask.any():
            _fuse_result = _topk_trash_fuse(
                all_feats=pool_feats_cat,
                all_pos=pool_gidx_cat,       # (P,) 1D global indices
                all_imp=pool_imp_cat,
                anchor_indices=anchor_indices,
                all_frame_ids=pool_frame_ids_t,
                top_k=topk_fusion,
                trash_ratio=trash_ratio,
                cross_frame_lambda=cross_frame_lambda,
                fusion_temperature=fusion_temperature,
                fusion_method=fusion_method,
                anchor_weight=anchor_weight,
                return_residual_info=_need_residual,
            )
            if _need_residual:
                fused_feat, fused_gidx, residual_info = _fuse_result
            else:
                fused_feat, fused_gidx = _fuse_result
        else:
            fused_feat = pool_feats_cat[anchor_indices].float()
            fused_gidx = pool_gidx_cat[anchor_indices]

        fused_feat = fused_feat.to(dtype)

        # ---- Pre-split: trash pool 聚类补充 (v8) ----
        if _want_residual and residual_info is not None and residual_info['count'] > 0:
            _res_tw = residual_info['trash_weights']
            _res_df = residual_info['drop_feats']
            _res_dp = residual_info['drop_pos']        # (N_drop,) 1D global indices
            _res_di = residual_info['drop_imp']
            _res_dfid = residual_info['drop_frame_ids']
            _res_dmi = residual_info['drop_mask_indices']

            _valid_res = (_res_tw > 0.01)
            # _valid_res = (_res_tw > (trash_ratio / topk_fusion))
            _n_valid = int(_valid_res.sum().item())

            if _n_valid > 0 and _n_valid >= K_residual * min_compression_ratio:
                _vr_feats = _res_tw[_valid_res].unsqueeze(-1) * _res_df[_valid_res]
                _vr_imp = _res_tw[_valid_res] * _res_di[_valid_res]
                if anchor_weight >= 1.0 and res_imp_alpha < 1.0:
                    _vr_imp = _vr_imp ** res_imp_alpha
                _vr_gidx = _res_dp[_valid_res]         # (N_valid,) 1D indices
                _vr_fid = _res_dfid[_valid_res]

                K_replace = min(K_residual, _n_valid)

                if res_method == 'dpc_knn':
                    replace_feats, _replace_center_idx = _dpc_knn_cluster(
                        features=_vr_feats,
                        num_clusters=K_replace,
                        k=min(7, _n_valid - 1), importance=_vr_imp,
                    )
                    replace_gidx = _vr_gidx[_replace_center_idx]
                else:
                    _replace_anc_idx = _dpp_select(_vr_feats, _vr_imp, K_replace)
                    replace_feats, replace_gidx = _topk_trash_fuse(
                        all_feats=_vr_feats,
                        all_pos=_vr_gidx,       # (N_valid,) 1D
                        all_imp=_vr_imp,
                        anchor_indices=_replace_anc_idx,
                        all_frame_ids=_vr_fid,
                        top_k=topk_fusion,
                        trash_ratio=0,
                        cross_frame_lambda=cross_frame_lambda,
                        fusion_temperature=fusion_temperature,
                        fusion_method=fusion_method,
                        anchor_weight=anchor_weight,
                    )

                # 直接 append 到 fused_feat / fused_gidx 末尾
                fused_feat = torch.cat([fused_feat, replace_feats.to(dtype)], dim=0)
                fused_gidx = torch.cat([fused_gidx, replace_gidx], dim=0)

                # _td_logger.info(
                #     f"[Compress] Group {g_idx}: pre-split append "
                #     f"K_group={K_group}, K_anchor={K_anchor}, K_residual={K_replace}, "
                #     f"trash_pool={_n_valid}/{residual_info['count']}, "
                #     f"min_ratio={min_compression_ratio}"
                # )
            else:
                # trash pool 不够大: 把 K_residual 预算还给 DPP
                if _n_valid > 0:
                    _td_logger.info(
                        f"[Compress] Group {g_idx}: trash pool too small for append "
                        f"(trash={_n_valid}, need>={K_residual * min_compression_ratio}), "
                        f"fallback to full K_group DPP"
                    )
                else:
                    _td_logger.info(
                        f"[Compress] Group {g_idx}: trash pool empty, fallback to full K_group DPP"
                    )
                # Fallback: 用 K_group 重新选 anchor 并融合
                K_anchor_fb = K_group
                if init_selected is not None and init_selected.shape[0] >= K_anchor_fb:
                    anchor_indices_fb = init_selected[:K_anchor_fb]
                elif anchor_method == "dpc_knn":
                    _, anchor_indices_fb = _dpc_knn_cluster(
                        features=pool_feats_cat, num_clusters=K_anchor_fb,
                        k=min(7, P - 1), importance=pool_imp_cat,
                    )
                    if init_selected is not None and init_selected.numel() > 0:
                        _existing = set(anchor_indices_fb.tolist())
                        _extra = [idx for idx in init_selected.tolist() if idx not in _existing]
                        if _extra:
                            _anc_imp = pool_imp_cat[anchor_indices_fb]
                            n_replace_fb = min(len(_extra), K_anchor_fb)
                            _, _weakest = _anc_imp.topk(n_replace_fb, largest=False)
                            for _r_idx, _e_idx in enumerate(_extra[:n_replace_fb]):
                                anchor_indices_fb[_weakest[_r_idx]] = _e_idx
                elif anchor_method == "facility_location":
                    anchor_indices_fb = _facility_location_select(
                        features=pool_feats_cat, importance=pool_imp_cat,
                        k=K_anchor_fb, init_selected=init_selected,
                    )
                elif anchor_method == "facility_location_optimized":
                    anchor_indices_fb = _facility_location_select_optimized(
                        features=pool_feats_cat, importance=pool_imp_cat,
                        k=K_anchor_fb, init_selected=init_selected,
                    )
                elif anchor_method == "leverage":
                    q = min(4 * K_anchor_fb, P, pool_feats_cat.shape[1])
                    U, S_val, V = torch.svd_lowrank(pool_feats_cat.float(), q=q)
                    leverage = (U ** 2).sum(dim=1)
                    score = leverage * pool_imp_cat
                    if init_selected is not None:
                        score[init_selected] = -1
                    _, extra = score.topk(K_anchor_fb - (len(init_selected) if init_selected is not None else 0))
                    anchor_indices_fb = torch.cat([init_selected, extra]) if init_selected is not None else extra
                elif anchor_method == "dpp_optimized":
                    anchor_indices_fb = _dpp_select_optimized(pool_feats_cat, pool_imp_cat, K_anchor_fb,
                                                 init_selected=init_selected)
                else:
                    anchor_indices_fb = _dpp_select(pool_feats_cat, pool_imp_cat, K_anchor_fb,
                                                 init_selected=init_selected)

                anchor_mask_fb = torch.zeros(P, dtype=torch.bool, device=device)
                anchor_mask_fb[anchor_indices_fb] = True
                drop_mask_fb = ~anchor_mask_fb

                if drop_mask_fb.any():
                    fused_feat, fused_gidx = _topk_trash_fuse(
                        all_feats=pool_feats_cat,
                        all_pos=pool_gidx_cat,
                        all_imp=pool_imp_cat,
                        anchor_indices=anchor_indices_fb,
                        all_frame_ids=pool_frame_ids_t,
                        top_k=topk_fusion,
                        trash_ratio=trash_ratio,
                        cross_frame_lambda=cross_frame_lambda,
                        fusion_temperature=fusion_temperature,
                        fusion_method=fusion_method,
                        anchor_weight=anchor_weight,
                        return_residual_info=False,
                    )
                else:
                    fused_feat = pool_feats_cat[anchor_indices_fb].float()
                    fused_gidx = pool_gidx_cat[anchor_indices_fb]
                fused_feat = fused_feat.to(dtype)
                anchor_indices = anchor_indices_fb

        all_fused_feats.append(fused_feat)
        all_fused_indices.append(fused_gidx)

    compressed_embeds = torch.cat(all_fused_feats, dim=0)
    keep_visual_indices = torch.cat(all_fused_indices, dim=0)

    # 按全局索引排序 (保持时间顺序)
    sort_order = keep_visual_indices.argsort()
    compressed_embeds = compressed_embeds[sort_order]
    keep_visual_indices = keep_visual_indices[sort_order]

    actual_total = compressed_embeds.shape[0]
    _td_logger.info(
        f"[Compress] Done: {T*N} → {actual_total} tokens "
        f"(budget={budget}, groups={len(groups)}, method=DPP+TopK)"
    )

    return compressed_embeds, keep_visual_indices


# ============================================================================
# LLaVA OneVision: LLM 内部 attention-based 剪枝 (1D RoPE)
# ============================================================================


def query_guided_pruning_llava(
    hidden_states: Tensor,       # (B, S, D)
    visual_token_range: tuple,   # (start, end)
    target_budget: int,          # 最终保留的视觉 token 数
    attention_mask: Optional[Tensor] = None,
    position_ids: Optional[Tensor] = None,
    cache_position: Optional[Tensor] = None,
    past_key_values=None,
    td_config=None,
    decoder_layer=None,
    position_embeddings=None,
    attn_history=None,           # list of (B, N_vis) tensors from trend observe layers
    attn_scores: Optional[Tensor] = None,  # (B, V) 由 decoder layer output_attentions 产出
) -> tuple:
    """LLM 内部 attention-score-based 剪枝 (LLaVA 版)。

    与 Qwen2.5-VL 版的区别:
      - position_ids: 2D (B, S) 而非 3D (B, 3, S)
      - 无 text_position_ids
      - 无 M-RoPE section 处理
      - cos/sin: (B, S, head_dim) 标准 1D RoPE

    当 attn_scores 不为 None 时, 直接使用 (output_attentions 方案);
    否则走原来的手动 q_proj/k_proj 计算逻辑 (向后兼容)。

    Returns:
        (hidden_states, attention_mask, position_ids,
         cache_position, position_embeddings, num_pruned, keep_idx)
    """
    B, S, D_h = hidden_states.shape
    v_start, v_end = visual_token_range
    num_visual = v_end - v_start

    if v_end > S or v_start >= S or num_visual <= 0:
        return hidden_states, attention_mask, position_ids, cache_position, position_embeddings, 0, None

    if num_visual <= target_budget:
        return hidden_states, attention_mask, position_ids, cache_position, position_embeddings, 0, None

    prune_method = getattr(td_config, 'llm_prune_method', 'text_token') if td_config else 'text_token'

    # ========================================================================
    # 优先使用外部传入的 attn_scores (output_attentions 方案)
    # ========================================================================
    if attn_scores is not None:
        scores = attn_scores  # (B, V) — 已由 decoder layer 正确计算
        # _td_logger.info(
        #     f"[QueryPrune-LLaVA] Using decoder-layer attn_scores: "
        #     f"[{scores.min().item():.6f}, {scores.max().item():.6f}]"
        # )
    elif decoder_layer is not None and position_embeddings is not None:
        self_attn = decoder_layer.self_attn
        head_dim = self_attn.head_dim
        num_heads = self_attn.config.num_attention_heads
        num_kv_heads = self_attn.config.num_key_value_heads
        num_kv_groups = num_heads // num_kv_heads

        with torch.no_grad():
            if prune_method == "text_token":
                text_prefix_h = hidden_states[:, :v_start, :]
                text_suffix_h = hidden_states[:, v_end:, :]
                if text_prefix_h.shape[1] + text_suffix_h.shape[1] == 0:
                    return hidden_states, attention_mask, position_ids, cache_position, position_embeddings, 0, None
                query_hidden = torch.cat([text_prefix_h, text_suffix_h], dim=1)
            else:
                query_hidden = hidden_states

            visual_hidden = hidden_states[:, v_start:v_end, :]

            q_states = self_attn.q_proj(query_hidden)
            k_states = self_attn.k_proj(visual_hidden)

            q_len = q_states.shape[1]
            v_len = k_states.shape[1]
            q_states = q_states.view(B, q_len, num_heads, head_dim).transpose(1, 2)
            k_states = k_states.view(B, v_len, num_kv_heads, head_dim).transpose(1, 2)

            # Apply RoPE
            cos, sin = position_embeddings
            if cos.ndim == 3:
                if prune_method == "text_token":
                    q_cos = torch.cat([cos[:, :v_start, :], cos[:, v_end:, :]], dim=1)
                    q_sin = torch.cat([sin[:, :v_start, :], sin[:, v_end:, :]], dim=1)
                else:
                    q_cos = cos
                    q_sin = sin
                k_cos = cos[:, v_start:v_end, :]
                k_sin = sin[:, v_start:v_end, :]
            elif cos.ndim == 2:
                if prune_method == "text_token":
                    q_cos = torch.cat([cos[:v_start, :], cos[v_end:, :]], dim=0).unsqueeze(0)
                    q_sin = torch.cat([sin[:v_start, :], sin[v_end:, :]], dim=0).unsqueeze(0)
                else:
                    q_cos = cos.unsqueeze(0)
                    q_sin = sin.unsqueeze(0)
                k_cos = cos[v_start:v_end, :].unsqueeze(0)
                k_sin = sin[v_start:v_end, :].unsqueeze(0)
            else:
                q_cos = k_cos = cos
                q_sin = k_sin = sin

            # GQA: repeat K
            from transformers.models.qwen2.modeling_qwen2 import repeat_kv as _repeat_kv
            if num_kv_groups > 1:
                k_states = _repeat_kv(k_states, num_kv_groups)

            scale = 1.0 / math.sqrt(head_dim)
            attn_logits = torch.matmul(q_states, k_states.transpose(-2, -1)) * scale
            attn_weights = attn_logits.softmax(dim=-1)

            # max over query, mean over heads
            scores = attn_weights.max(dim=2).values.mean(dim=1)  # (B, V)

        # _td_logger.info(
        #     f"[QueryPrune] method={prune_method}, "
        #     f"Q=({q_len},), K=({v_len},), "
        #     f"scores: [{scores.min().item():.6f}, {scores.max().item():.6f}]"
        # )
    else:
        # Fallback: cosine similarity
        _td_logger.warning("[QueryPrune] fallback to cosine similarity")
        visual_hidden = hidden_states[:, v_start:v_end, :]
        text_all = torch.cat([hidden_states[:, :v_start, :], hidden_states[:, v_end:, :]], dim=1)
        if text_all.shape[1] == 0:
            return hidden_states, attention_mask, position_ids, cache_position, position_embeddings, 0, None
        query = text_all.mean(dim=1, keepdim=True)
        query_normed = F.normalize(query, dim=-1)
        visual_normed = F.normalize(visual_hidden, dim=-1)
        scores = (visual_normed * query_normed).sum(dim=-1)

    # ========================================================================
    # 融合 attn_history (Trend Observe 多层观测分数)
    # ========================================================================
    if attn_history is not None and len(attn_history) > 0:
        _trend_weight = getattr(td_config, 'trend_weight', 0.3) if td_config else 0.3
        _history_stack = torch.stack(attn_history, dim=0)  # (L, B, N_vis)
        _history_mean = _history_stack.mean(dim=0)  # (B, N_vis)
        # 归一化到 [0,1]
        _h_min = _history_mean.min(dim=-1, keepdim=True).values
        _h_max = _history_mean.max(dim=-1, keepdim=True).values
        _h_range = (_h_max - _h_min).clamp(min=1e-8)
        _history_norm = (_history_mean - _h_min) / _h_range

        _s_min = scores.min(dim=-1, keepdim=True).values
        _s_max = scores.max(dim=-1, keepdim=True).values
        _s_range = (_s_max - _s_min).clamp(min=1e-8)
        _scores_norm = (scores - _s_min) / _s_range

        scores = (1 - _trend_weight) * _scores_norm + _trend_weight * _history_norm
        # _td_logger.info(
        #     f"[QueryPrune] Fused {len(attn_history)} trend layers (weight={_trend_weight}), "
        #     f"final scores: [{scores.min().item():.6f}, {scores.max().item():.6f}]"
        # )

    # Top-k selection
    k = min(target_budget, num_visual)
    _, top_indices = scores.topk(k, dim=1)
    top_indices = top_indices.sort(dim=1).values
    keep_idx = top_indices[0]

    kept_visual = hidden_states[0, v_start + keep_idx, :].unsqueeze(0)
    text_prefix = hidden_states[:, :v_start, :]
    text_suffix = hidden_states[:, v_end:, :]
    new_hidden = torch.cat([text_prefix, kept_visual, text_suffix], dim=1)

    # Update attention_mask
    new_attention_mask = attention_mask
    if attention_mask is not None and attention_mask.ndim == 2:
        prefix_mask = attention_mask[:, :v_start]
        suffix_mask = attention_mask[:, v_end:]
        visual_mask = torch.ones(B, k, device=attention_mask.device, dtype=attention_mask.dtype)
        new_attention_mask = torch.cat([prefix_mask, visual_mask, suffix_mask], dim=1)

    # Update position_ids (1D RoPE)
    new_position_ids = position_ids
    if position_ids is not None:
        if position_ids.ndim == 2:
            # Standard 1D: (B, S)
            prefix_pos = position_ids[:, :v_start]
            suffix_pos = position_ids[:, v_end:]
            visual_pos = position_ids[:, v_start:v_end]
            kept_pos = visual_pos[:, keep_idx]
            new_position_ids = torch.cat([prefix_pos, kept_pos, suffix_pos], dim=1)
        elif position_ids.ndim == 3:
            # Safety: 3D M-RoPE (shouldn't happen for LLaVA, but handle gracefully)
            prefix_pos = position_ids[:, :, :v_start]
            suffix_pos = position_ids[:, :, v_end:]
            visual_pos = position_ids[:, :, v_start:v_end]
            kept_pos = visual_pos[:, :, keep_idx]
            new_position_ids = torch.cat([prefix_pos, kept_pos, suffix_pos], dim=2)

    # Update cache_position
    new_cache_position = cache_position
    if cache_position is not None:
        new_cache_position = torch.arange(0, new_hidden.shape[1], device=cache_position.device)

    # Update position_embeddings
    new_pos_emb = position_embeddings
    if position_embeddings is not None:
        cos, sin = position_embeddings
        # Rebuild by keeping non-visual + kept visual + suffix
        keep_global = torch.cat([
            torch.arange(v_start, device=keep_idx.device),
            v_start + keep_idx,
            torch.arange(v_end, S, device=keep_idx.device),
        ])
        if cos.ndim == 3:
            new_cos = cos[:, keep_global, :]
            new_sin = sin[:, keep_global, :]
        elif cos.ndim == 2:
            new_cos = cos[keep_global, :]
            new_sin = sin[keep_global, :]
        else:
            new_cos = cos
            new_sin = sin
        new_pos_emb = (new_cos, new_sin)

    # Clean KV cache
    if past_key_values is not None and hasattr(past_key_values, 'key_cache'):
        _kv_keep = torch.cat([
            torch.arange(v_start, device=keep_idx.device),
            v_start + keep_idx,
            torch.arange(v_end, S, device=keep_idx.device),
        ])
        for _kv_layer_idx in range(len(past_key_values.key_cache)):
            if past_key_values.key_cache[_kv_layer_idx].numel() > 0:
                kv_len = past_key_values.key_cache[_kv_layer_idx].shape[2]
                if kv_len == S:
                    past_key_values.key_cache[_kv_layer_idx] = past_key_values.key_cache[_kv_layer_idx].index_select(2, _kv_keep)
                    past_key_values.value_cache[_kv_layer_idx] = past_key_values.value_cache[_kv_layer_idx].index_select(2, _kv_keep)

    num_pruned = num_visual - k
    _td_logger.info(f"[QueryPrune] {num_visual} → {k} visual tokens (pruned {num_pruned})")

    return new_hidden, new_attention_mask, new_position_ids, new_cache_position, new_pos_emb, num_pruned, keep_idx


def _deepstack_process(
    hidden_states: torch.Tensor,
    visual_pos_masks: torch.Tensor,
    deepstack_embed: torch.Tensor,
) -> torch.Tensor:
    """DeepStack 残差注入: 在视觉位置做 hidden_states += deepstack_embed。

    Args:
        hidden_states: (B, S, D)
        visual_pos_masks: (B, S) bool mask, True = 视觉 token
        deepstack_embed: (num_visual, D) 该层对应的 DeepStack 特征
    """
    hidden_states = hidden_states.clone()
    hidden_states[visual_pos_masks] = hidden_states[visual_pos_masks] + deepstack_embed
    return hidden_states