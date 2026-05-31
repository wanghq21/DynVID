"""
新增两种语义分割方法: "pca" 和 "attention"
输出格式与 felzenszwalb 一致: seg_labels_list + K_raw_list
分割后沿用 softmax 加权融合管线。
"""

import numpy as np
from typing import List, Tuple


# ============================================================================
#  通用: 4-连通 Union-Find 连通分量
# ============================================================================

def _cc_from_adj_mask(N: int, frame_h: int, frame_w: int,
                      edge_mask_right: np.ndarray,
                      edge_mask_down: np.ndarray) -> np.ndarray:
    """根据 4-连通边的布尔掩码做连通分量标记。

    Args:
        N:               token 总数 (= frame_h * frame_w)
        frame_h, frame_w: 帧的 token 网格尺寸
        edge_mask_right:  shape=(frame_h, frame_w-1)  右向边是否连通
        edge_mask_down:   shape=(frame_h-1, frame_w)  下向边是否连通

    Returns:
        labels: shape=(N,)  连通分量 id (从 0 开始, 连续编号)
    """
    parent = list(range(N))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(frame_h):
        for j in range(frame_w - 1):
            if edge_mask_right[i, j]:
                union(i * frame_w + j, i * frame_w + j + 1)

    for i in range(frame_h - 1):
        for j in range(frame_w):
            if edge_mask_down[i, j]:
                union(i * frame_w + j, (i + 1) * frame_w + j)

    root_map = {}
    labels = np.empty(N, dtype=np.int32)
    next_id = 0
    for idx in range(N):
        r = find(idx)
        if r not in root_map:
            root_map[r] = next_id
            next_id += 1
        labels[idx] = root_map[r]

    return labels




def _otsu_threshold(values: np.ndarray) -> float:
    """Otsu's method: 在一维数值数组上找到最优二分阈值。

    最大化类间方差 (inter-class variance), 将值分成"连通"和"断开"两类。
    - 简单帧: 相似度分布单峰 -> Otsu 阈值偏低 -> 大部分边保留 -> 少 segment
    - 复杂帧: 相似度分布双峰 -> Otsu 阈值在谷底 -> 自然分离 -> 多 segment

    复杂度 O(N + B) 其中 B=256 为直方图 bin 数, 非常快。
    """
    # 量化到 256 bins
    vmin, vmax = float(values.min()), float(values.max())
    if vmax - vmin < 1e-10:
        return float(vmin)

    n_bins = 256
    hist, bin_edges = np.histogram(values, bins=n_bins, range=(vmin, vmax))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    total = hist.sum()
    if total == 0:
        return float((vmin + vmax) / 2)

    sum_total = (hist * bin_centers).sum()
    sum_bg = 0.0
    weight_bg = 0
    max_variance = 0.0
    best_threshold = float(bin_centers[0])

    for i in range(n_bins):
        weight_bg += hist[i]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += hist[i] * bin_centers[i]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance > max_variance:
            max_variance = variance
            best_threshold = float(bin_centers[i])

    return best_threshold


# ============================================================================
#  方法 1: PCA + 位置编码 + 阈值连通分量
# ============================================================================

def _pca_cc_segment(
    frame_feats_np: np.ndarray,
    frame_h: int,
    frame_w: int,
    n_components: int = 8,
    pos_weight: float = 0.5,
    alpha: float = 0.5,
    min_segments: int = 2,
) -> Tuple[np.ndarray, int]:
    """单帧 PCA + 阈值连通分量分割。

    Pipeline:
      1. PCA 降维到 n_components 维 -> 放大语义主成分, 抑制噪声维度
      2. L2 归一化
      3. 拼接归一化空间坐标 (行/列 in [0,1]) * pos_weight -> 保留位置先验
      4. 再次 L2 归一化
      5. 向量化计算 4-连通邻居对的余弦相似度
      6. Otsu 自适应阈值, alpha 为缩放因子
         Otsu 根据相似度分布自动找最优分割点
         简单帧 (单峰) -> 低阈值 -> 少 segment; 复杂帧 (双峰) -> 高阈值 -> 多 segment
         alpha 缩放 Otsu 阈值: <1 更粗, =1 原始, >1 更细
      7. sim > threshold 的边保留 -> 4-连通连通分量 = 语义块

    Args:
        frame_feats_np: (N, D) 单帧 token 特征 (ViT hidden states)
        frame_h, frame_w: token 网格尺寸, N = frame_h * frame_w
        n_components: PCA 目标维度, 默认 8
        pos_weight:   位置编码权重, 默认 0.5
                      0 = 不用位置 (纯语义分割)
                      >0 = 引入位置偏置, 鼓励空间相邻 token 归为一组
        alpha: Otsu 阈值缩放因子, 默认 0.5
               < 1.0 = 在 Otsu 阈值基础上降低 -> 更多边保留 -> 粗分割 (少 segment)
               = 1.0 = 使用 Otsu 原始阈值
               > 1.0 = 在 Otsu 阈值基础上提高 -> 更多边断开 -> 细分割 (多 segment)

    Returns:
        labels: (N,) segment 标签, 从 0 开始连续编号
        K: segment 数量
    """
    N, D = frame_feats_np.shape
    assert N == frame_h * frame_w

    # Step 1: PCA 降维 (手动实现, 避免 sklearn 依赖)
    n_comp = min(n_components, N - 1, D)
    feats = frame_feats_np.astype(np.float32)
    mean = feats.mean(axis=0)
    feats_centered = feats - mean

    if N < D:
        cov = feats_centered @ feats_centered.T  # (N, N)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1][:n_comp]
        U = eigvecs[:, idx]
        feats_pca = U * np.sqrt(np.maximum(eigvals[idx], 0))
    else:
        cov = feats_centered.T @ feats_centered / N  # (D, D)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1][:n_comp]
        V = eigvecs[:, idx]
        feats_pca = feats_centered @ V

    # Step 2: L2 归一化
    norms = np.linalg.norm(feats_pca, axis=1, keepdims=True)
    feats_pca = feats_pca / np.maximum(norms, 1e-8)

    # Step 3: 拼接位置编码
    if pos_weight > 0:
        rows = np.arange(frame_h, dtype=np.float32).reshape(-1, 1).repeat(
            frame_w, axis=1).reshape(-1) / max(frame_h - 1, 1)
        cols = np.arange(frame_w, dtype=np.float32).reshape(1, -1).repeat(
            frame_h, axis=0).reshape(-1) / max(frame_w - 1, 1)
        pos_feats = np.stack([rows, cols], axis=1) * pos_weight  # (N, 2)
        feats_combined = np.concatenate([feats_pca, pos_feats], axis=1)
        # Step 4: 再次 L2 归一化
        norms = np.linalg.norm(feats_combined, axis=1, keepdims=True)
        feats_combined = feats_combined / np.maximum(norms, 1e-8)
    else:
        feats_combined = feats_pca

    # Step 5: 向量化计算 4-连通邻居余弦相似度
    feats_2d = feats_combined.reshape(frame_h, frame_w, -1)

    sim_right = np.sum(
        feats_2d[:, :-1, :] * feats_2d[:, 1:, :], axis=-1
    )  # (frame_h, frame_w-1)

    sim_down = np.sum(
        feats_2d[:-1, :, :] * feats_2d[1:, :, :], axis=-1
    )  # (frame_h-1, frame_w)

    # Step 6: Otsu 自适应阈值 (根据内容复杂度自动调整)
    all_sims = np.concatenate([sim_right.ravel(), sim_down.ravel()])
    otsu_thr = _otsu_threshold(all_sims)
    # alpha 作为缩放因子: threshold = otsu * alpha
    # alpha < 1 -> 阈值降低 -> 保留更多边 -> 粗分割
    # alpha > 1 -> 阈值升高 -> 断开更多边 -> 细分割
    threshold = otsu_thr * alpha

    # Step 7: 连通分量 (sim > threshold 表示连通)
    edge_mask_right = sim_right > threshold
    edge_mask_down = sim_down > threshold

    labels = _cc_from_adj_mask(N, frame_h, frame_w,
                               edge_mask_right, edge_mask_down)
    K = int(labels.max()) + 1

    return labels, K


def pca_cc_segment_all_frames(
    features_np: np.ndarray,
    total_T: int,
    N_merged: int,
    frame_h: int,
    frame_w: int,
    n_components: int = 8,
    pos_weight: float = 0.5,
    alpha: float = 0.5,
) -> Tuple[List[np.ndarray], List[int]]:
    """对所有帧做 PCA + 阈值连通分量分割。

    Args:
        features_np: (total_T * N_merged, D) 所有帧的 token 特征
        total_T: 帧数
        N_merged: 每帧 token 数
        frame_h, frame_w: 每帧 token 网格尺寸
        n_components, pos_weight, alpha: 见 _pca_cc_segment

    Returns:
        seg_labels_list: List[np.ndarray], 长度 = total_T
        K_raw_list: List[int], 长度 = total_T
    """
    seg_labels_list = []
    K_raw_list = []

    for t in range(total_T):
        frame_feats = features_np[t * N_merged: (t + 1) * N_merged]
        labels, K = _pca_cc_segment(
            frame_feats, frame_h, frame_w,
            n_components=n_components,
            pos_weight=pos_weight,
            alpha=alpha,
        )
        seg_labels_list.append(labels)
        K_raw_list.append(K)

    return seg_labels_list, K_raw_list


# ============================================================================
#  方法 2: Attention Map + 阈值连通分量
# ============================================================================

def _attention_cc_segment(
    attn_affinity_np: np.ndarray,
    frame_h: int,
    frame_w: int,
    alpha: float = 0.5,
    min_segments: int = 2,
) -> Tuple[np.ndarray, int]:
    """单帧 Attention Map + 阈值连通分量分割。

    Pipeline:
      1. 输入: (N, N) 帧内 attention 亲和度矩阵 (已含 RoPE 位置信息)
         - 由 ViT self-attention 的 softmax(Q@K^T / sqrt(d)) 计算
         - 多头平均后得到
         - RoPE 编码了每个 token 的 (t, h, w) 3D 位置
      2. 对称化: A = (A + A^T) / 2, L2 行归一化
      3. 向量化计算 4-连通邻居的 attention 行余弦相似度 cosine(A[i,:], A[j,:])
         (衡量"两个 token 是否关注相同的目标", 不受 RoPE 近邻偏置影响)
      4. Otsu 自适应阈值, alpha 为缩放因子
      5. sim > threshold -> 连通 -> 连通分量 = 语义块

    为什么 Attention 行相似度比原始特征余弦距离更适合语义分割:
      - Attention 经过 Q, K 线性投影, 学到了"什么应该关注什么"的模式
      - 使用行余弦相似度 cosine(A[i,:], A[j,:]) 而非 raw A[i,j]
        避免 RoPE 近邻偏置 (raw A[i,j] 对所有邻居都高, 无法区分)
      - "是否关注相同目标" 是真正的语义信号: 同实体内 token 注意力模式高度一致
      - 余弦距离在深层特征上只有 ~1.1x 的比值, 几乎不可区分

    Args:
        attn_affinity_np: (N, N) 帧内多头平均 attention 矩阵
        frame_h, frame_w: token 网格尺寸
        alpha: 分割粒度, 默认 0.5

    Returns:
        labels: (N,) segment 标签
        K: segment 数量
    """
    N = frame_h * frame_w
    assert attn_affinity_np.shape == (N, N), (
        f"Expected ({N}, {N}), got {attn_affinity_np.shape}"
    )

    # Step 1: 对称化 + L2 行归一化
    # 对称化保证 A[i,:] 和 A[j,:] 在同一空间
    A = (attn_affinity_np + attn_affinity_np.T).astype(np.float32) / 2.0
    # L2 行归一化: 让后续点积 = 余弦相似度
    row_norms = np.linalg.norm(A, axis=1, keepdims=True)
    A_normed = A / np.maximum(row_norms, 1e-8)

    # Step 2: 向量化计算 4-连通邻居的 attention 行余弦相似度
    # cosine(A[i,:], A[j,:]) = "两个 token 是否关注相同的目标"
    # - 同语义实体: 注意力模式相似 -> 高相似度
    # - 跨语义实体: 注意力模式不同 -> 低相似度
    # 这避免了 RoPE 近邻偏置 (raw A[i,j] 对邻居总是很高)
    A_2d = A_normed.reshape(frame_h, frame_w, N)

    # 水平邻居余弦相似度 (frame_h, frame_w-1)
    sim_right = np.sum(
        A_2d[:, :-1, :] * A_2d[:, 1:, :], axis=-1
    )
    # 垂直邻居余弦相似度 (frame_h-1, frame_w)
    sim_down = np.sum(
        A_2d[:-1, :, :] * A_2d[1:, :, :], axis=-1
    )

    # Step 3: Otsu 自适应阈值
    all_sims = np.concatenate([sim_right.ravel(), sim_down.ravel()])
    otsu_thr = _otsu_threshold(all_sims)
    threshold = otsu_thr * alpha

    # Step 4: 连通分量 (sim > threshold 表示连通)
    edge_mask_right = sim_right > threshold
    edge_mask_down = sim_down > threshold

    labels = _cc_from_adj_mask(N, frame_h, frame_w,
                               edge_mask_right, edge_mask_down)
    K = int(labels.max()) + 1

    return labels, K


def attention_cc_segment_all_frames(
    attn_per_frame: List[np.ndarray],
    frame_h: int,
    frame_w: int,
    alpha: float = 0.5,
) -> Tuple[List[np.ndarray], List[int]]:
    """对所有帧做 Attention + 阈值连通分量分割。

    Args:
        attn_per_frame: List[(N, N)], 每帧的 attention 亲和度矩阵
        frame_h, frame_w: 每帧 token 网格尺寸
        alpha: 见 _attention_cc_segment

    Returns:
        seg_labels_list: List[np.ndarray]
        K_raw_list: List[int]
    """
    seg_labels_list = []
    K_raw_list = []

    for t, attn_frame in enumerate(attn_per_frame):
        labels, K = _attention_cc_segment(
            attn_frame, frame_h, frame_w,
            alpha=alpha,
        )
        seg_labels_list.append(labels)
        K_raw_list.append(K)

    return seg_labels_list, K_raw_list


# ============================================================================
#  Attention 权重提取辅助函数 (在 ViT Attention Forward 中调用)
# ============================================================================

def compute_and_save_attn_weights(q, k, td_config, grid_thw_info, scale=None):
    """从 Q, K (RoPE 后) 计算逐帧 intra-frame attention 并保存到 td_config。

    设计原则:
      - 只计算帧内 (intra-frame) attention, 复杂度 O(T * N^2 * d)
      - 多头平均, 存为 List[(N, N)] numpy 数组
      - Q, K 已经过 RoPE, 天然包含 3D (t, h, w) 位置信息
      - 在 GPU 上计算, 结果转 CPU numpy 存储

    Args:
        q: (batch, num_heads, seq_len, head_dim) -- RoPE 后的 query
        k: (batch, num_heads, seq_len, head_dim) -- RoPE 后的 key
        td_config: TensorDecompConfig 实例
        grid_thw_info: dict with 'total_T', 'N_merged'
        scale: attention scale, 默认 1/sqrt(head_dim)

    Side Effects:
        设置 td_config._vit_attn_per_frame = List[np.ndarray], 每个 (N, N)
    """
    import torch

    total_T = grid_thw_info['total_T']
    N = grid_thw_info['N_merged']
    if scale is None:
        scale = q.shape[-1] ** -0.5

    attn_per_frame = []

    with torch.no_grad():
        for t in range(total_T):
            start = t * N
            end = start + N
            q_t = q[:, :, start:end, :]  # (1, H, N, d)
            k_t = k[:, :, start:end, :]  # (1, H, N, d)

            # (1, H, N, N)
            attn_t = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale
            attn_t = torch.softmax(attn_t, dim=-1)

            # 多头平均 -> (N, N)
            attn_mean = attn_t.mean(dim=1).squeeze(0)
            attn_per_frame.append(attn_mean.cpu().float().numpy())

            del attn_t, attn_mean

    td_config._vit_attn_per_frame = attn_per_frame

# ============================================================================
# v11b: Compute per-token "attention received" score at fuse layer
# ============================================================================

def compute_and_save_attn_received(td_config, query_states, key_states, cu_seqlens):
    """Compute per-token attention received score from Q, K at the fuse layer.

    For each frame, computes the self-attention matrix from Q and K,
    then sums each column to get how much each token is "attended to"
    by all other tokens (FlashVID-style CLS attention proxy).

    Unlike compute_and_save_attn_weights which saves full (N,N) matrices,
    this only saves the (N,) received-attention vector per frame — much lighter.

    Args:
        td_config: TensorDecompConfig with _v21_grid_thw_info
        query_states: (seq_len, num_heads, head_dim) after RoPE
        key_states:   (seq_len, num_heads, head_dim) after RoPE
        cu_seqlens:   cumulative sequence lengths (not used, kept for interface compat)

    Side Effects:
        Sets td_config._vit_attn_received_raw = List[np.ndarray(N_merged,)]
    """
    import torch
    import numpy as np

    grid_info = getattr(td_config, '_v21_grid_thw_info', None)
    if grid_info is None:
        return
    total_T = grid_info['total_T']
    N = grid_info['N_merged']

    # query_states shape: (seq_len, num_heads, head_dim)
    head_dim = query_states.shape[-1]
    scale = head_dim ** -0.5

    num_heads = query_states.shape[1]  # (seq_len, num_heads, head_dim)
    attn_received_list = []

    # OPT: mini-batch processing (reduces kernel launch overhead vs per-frame loop)
    BATCH = min(total_T, 4)

    with torch.no_grad():
        for b_start in range(0, total_T, BATCH):
            b_end = min(b_start + BATCH, total_T)
            B = b_end - b_start
            seq_start = b_start * N
            seq_end = b_end * N
            # (B*N, H, d) -> (B, N, H, d) -> (B, H, N, d)
            Q_b = query_states[seq_start:seq_end].reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
            K_b = key_states[seq_start:seq_end].reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
            # Batched attention: (B, H, N, N)
            attn_b = torch.matmul(Q_b, K_b.transpose(-2, -1)) * scale
            attn_b = torch.softmax(attn_b, dim=-1)
            # Mean across heads -> (B, N, N), column sum -> (B, N)
            recv_b = attn_b.mean(dim=1).sum(dim=1)  # (B, N)
            for i in range(B):
                attn_received_list.append(recv_b[i].cpu().float().numpy())
            del attn_b, recv_b, Q_b, K_b

    td_config._vit_attn_received_raw = attn_received_list

