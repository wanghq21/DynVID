"""
TensorDecompConfig: DySeg + DPP Anchor Selection + Top-K Soft Fusion 视频token压缩配置类。

Pipeline (post-projector):
  1. DySeg 分组: 相邻帧按余弦相似度阈值分组
  2. 组预算分配: 按 Effective Rank 或 MCD (Mean Cosine Distance) 分配每组 token 预算
  3. 组内 DPP (Determinantal Point Process) anchor 选择:
     importance = (1 - w_dyn - w_query) * cls_attn + w_dyn * dynamism + w_query * query_score
     kernel L[i][j] = importance(i) * sim(i,j) * importance(j)
     贪心最大化 det 增量 (Cholesky 增量更新, O(K*N))
  4. 组内 Top-K 软分配 + 自适应垃圾桶 Softmax 融合:
     每个 drop token 分配到 top-k 个最相似 anchor (非 argmax 硬分配)
     自适应垃圾桶: threshold = per_token_max_sim_among_drops * ratio
     冗余 drop token 倾向被丢弃, 独特 drop token 倾向被保留
  5. (可选) LLM 内部软剪枝: soft mask 引导注意力收敛
  6. LLM 内部硬剪枝: attention-score-based token 物理移除

所有信号 (CLS attention, dynamism) 在 projector 之后计算, ViT 完全不修改。
  7. (可选) ViT 内部 token 融合: 中层画圈 + 深层圈内融合, 减少送给 projector 的 token 数
"""


class TensorDecompConfig:
    """DySeg + DPP + Top-K Soft Fusion 视频token压缩配置类。

    Args:
        # ── Post-projector 压缩参数 ──
        stage2_retention_ratio (float): Post-projector 保留率。
            budget = T * N * stage2_retention_ratio。默认 0.20。
        dyseg_threshold (float): DySeg 分组的余弦相似度阈值。
            相邻帧相似度 > threshold → 同组。默认 0.85。
        cross_frame_lambda (float): 跨帧时间距离惩罚系数。
            sim_ij -= λ × |frame_i - frame_j|。
            inf = 帧内硬约束, 0 = 完全跨帧, >0 = 软惩罚。默认 0.0。
        fusion_temperature (float): Softmax fusion 温度参数。
            控制 softmax(cosine_sim / τ) 的尖锐度。默认 0.01。
        min_segment_num (int): DySeg 最少分段数, 0=不限制。默认 0。
        complementary_segment (bool): 不够 min_segment_num 时是否自动补刀。默认 True。

        # ── 组预算分配 ──
        group_budget_method (str): 组预算分配方法。
            "effective_rank": 有效秩 — SVD → 归一化奇异值 → exp(Shannon entropy)
                直接代表"需要多少个独立方向来表示这个组", 已包含组大小信息, 无需乘以 len(group)
            "mcd": Mean Cosine Distance — 1 - mean(pairwise_sim)
                需要乘以 len(group), 因为 MCD 本身是 scale-insensitive 的
            默认 "effective_rank"。
        effective_rank_k (int): effective_rank2 分支中 randomized SVD 的基础 q 系数。
            q = min(effective_rank_k * sqrt(len(group)), P, D)。默认 64。

        # ── DPP 重要性信号 ──
        anchor_method (str): anchor 选择方法。
            "dpp": Determinantal Point Process 贪心选择。
            "leverage": leverage score 选择。
            默认 "dpp"。
        w_dyn (float): dynamism 信号权重 (convex combination)。
            importance = (1 - w_dyn - w_query) * cls_attn + w_dyn * dynamism + w_query * query_score
            默认 0.3。
        w_query (float): query score 信号权重。默认 0.05。
            注意: w_dyn + w_query < 1.0, 否则 cls_attn 权重为负。
        asym_w (float): dynamism 中 asymmetry 的权重。
            dynamism = (1 - asym_w) * novelty + asym_w * asymmetry
            novelty = 1 - max(fwd_max, bwd_max)  → 前后都不相似 = 新内容
            asymmetry = |fwd_max - bwd_max|  → 前后不对称 = 运动方向变化
            默认 0.3。
        dynamism_window (int): 动态性计算的时间窗口大小。默认 1。
        cls_attn_method (str): CLS attention 信号的计算方式。
            "pseudo_cls" / "col_mean" / "l2_norm" / "adaptive"。默认 "pseudo_cls"。

        # ── Top-K 软分配 + 自适应垃圾桶 ──
        topk_fusion (int): 每个 drop token 分配到的 anchor 数量。
            1 = 退化为 argmax 硬分配, 越大越接近全局软分配。默认 3。
        trash_ratio (float): 自适应垃圾桶阈值比例。
            threshold = per_token_max_sim_among_drops * trash_ratio
            冗余 drop: max_sim 高 → threshold 高 → 容易被丢弃
            独特 drop: max_sim 低 → threshold 低 → 倾向保留
            0 = 无垃圾桶 (所有 drop 都被分配), >1 = 激进丢弃。默认 0.6。
        fusion_method (str): 融合方式。
            "mean": anchor-protected 均值融合, fused = α*anchor + (1-α)*drop_centroid
            "softmax": scatter 向量化 sim_to_centroid softmax 融合
            "softmax_imp": 和 softmax 一样但额外乘 importance
            默认 "mean"。
        anchor_weight (float): mean 融合中 anchor 的权重 α。
            fused = α * anchor + (1-α) * drop_centroid
            α=0.5 → anchor 贡献 50% (推荐), α=1.0 → 不融合
            仅影响 mean 方法, softmax/softmax_imp 有自己的 per-cluster normalization。
            默认 0.5。
        residual_ratio (float): 残留池预留预算比例 (v6)。
            K_anchor = round(K_group × (1 - residual_ratio)), K_residual = K_group - K_anchor。
            残留池收集每个 drop 的 w_trash × feat, DPP 选 K_residual 个补充 token。
            0 = 禁用残留池 (等价于 v5), 推荐 0.10~0.20。
            默认 0.0。
        res_imp_alpha (float): DPP₂ importance 压缩指数 (anchor_weight>=1.0 时生效)。默认 0.3。
        res_method (str): 残差池处理方式。
            "dpp": DPP 选择。
            "dpc_knn": DPC-KNN 聚类选择。
            默认 "dpc_knn"。
        min_compression_ratio (int): 先选后替策略的最小压缩比门槛。
            trash_pool_size >= K_replace_max × min_compression_ratio 时才执行替换。
            值越大越保守 (要求 trash pool 更大才替换)。默认 3。

        # ── LLM 内部剪枝 ──
        query_prune_layer (int): LLM 内部剪枝层号, -1=禁用。默认 21。
        llm_prune_ratio (float): LLM 剪枝保留比, 1.0=不额外剪枝。默认 1.0。
        llm_prune_method (str): "text_token" / "all_token"。默认 "text_token"。
        soft_prune_layer (int): 软剪枝层号, -1=禁用。默认 -1。

        # ── ViT 内部 token 融合 ──
        vit_fusion_enabled (bool): 总开关。默认 False。
        vit_fusion_group_layer (int): 画圈层号。默认 15。
        vit_fusion_merge_layer (int): 融合层号。默认 25。
        vit_fusion_sim_threshold (float): 画圈 cosine 相似度阈值。默认 0.85。
        vit_fusion_alpha (float): 均值偏移融合强度。默认 0.5。
        vit_fusion_fz_k (float): ViT fusion farthest zone k, -1=禁用。默认 -1.0。
        vit_fusion_seg_method (str): ViT fusion 分组方法。默认 "threshold_cc"。
        vit_fusion_merge_mode (str): ViT fusion 合并模式。默认 "mean"。
        vit_fusion_temperature (float): ViT fusion softmax 温度。默认 0.1。
    """

    VALID_PRUNE_METHODS = ("text_token", "all_token", "last_token")
    VALID_CLS_METHODS = ("pseudo_cls", "col_mean", "l2_norm", "adaptive", "col_mean_l2")
    VALID_GROUP_BUDGET_METHODS = ("effective_rank", "effective_rank2", "mcd", "dispersion", "erank_dispersion", "uniform", "moment_ratio")
    VALID_FUSION_METHODS = ("mean", "softmax", "softmax_imp")
    VALID_ANCHOR_METHODS = ("dpp", "leverage", "facility_location", "dpp_optimized", "facility_location_optimized", "dpc_knn")
    VALID_RES_METHODS = ("dpp", "dpc_knn")
    VALID_VIT_FUSION_SEG_METHODS = ("threshold_cc",)
    VALID_VIT_FUSION_MERGE_MODES = ("mean",)

    def __init__(
        self,
        # ---- Post-projector 压缩参数 ----
        stage2_retention_ratio: float = 0.20,
        dyseg_threshold: float = 0.85,
        cross_frame_lambda: float = 0.0,
        fusion_temperature: float = 0.01,
        min_segment_num: int = 0,
        complementary_segment: bool = True,
        # ---- 组预算分配 ----
        group_budget_method: str = "effective_rank",
        effective_rank_k: int = 64,
        # ---- DPP 重要性信号 ----
        anchor_method: str = "dpp",
        w_dyn: float = 0.3,
        w_query: float = 0.05,
        asym_w: float = 0.3,
        dynamism_window: int = 1,
        cls_attn_method: str = "pseudo_cls",
        # ---- Top-K 软分配 + 垃圾桶 ----
        topk_fusion: int = 3,
        trash_ratio: float = 0.6,
        fusion_method: str = "mean",
        anchor_weight: float = 0.5,
        residual_ratio: float = 0.0,
        res_imp_alpha: float = 0.3,
        res_method: str = "dpc_knn",
        min_compression_ratio: int = 1,
        # ---- LLM 内部剪枝 ----
        query_prune_layer: int = 21,
        llm_prune_ratio: float = 1.0,
        llm_prune_method: str = "text_token",
        soft_prune_layer: int = -1,
        # ---- ViT 内部 token 融合 ----
        vit_fusion_enabled: bool = False,
        vit_fusion_group_layer: int = 15,
        vit_fusion_merge_layer: int = 25,
        vit_fusion_sim_threshold: float = 0.85,
        vit_fusion_alpha: float = 0.5,
        vit_fusion_fz_k: float = -1.0,
        vit_fusion_seg_method: str = "threshold_cc",
        vit_fusion_merge_mode: str = "mean",
        vit_fusion_temperature: float = 0.1,
    ):
        # Post-projector 压缩
        self.stage2_retention_ratio = stage2_retention_ratio
        self.dyseg_threshold = dyseg_threshold
        self.cross_frame_lambda = cross_frame_lambda
        self.fusion_temperature = fusion_temperature
        self.min_segment_num = min_segment_num
        self.complementary_segment = complementary_segment

        # 组预算分配
        assert group_budget_method in self.VALID_GROUP_BUDGET_METHODS, (
            f"group_budget_method must be one of {self.VALID_GROUP_BUDGET_METHODS}, "
            f"got '{group_budget_method}'"
        )
        self.group_budget_method = group_budget_method
        self.effective_rank_k = effective_rank_k

        # DPP 重要性信号
        assert anchor_method in self.VALID_ANCHOR_METHODS, (
            f"anchor_method must be one of {self.VALID_ANCHOR_METHODS}, "
            f"got '{anchor_method}'"
        )
        self.anchor_method = anchor_method
        assert w_dyn + w_query < 1.0 + 1e-6, (
            f"w_dyn + w_query must be < 1.0, got w_dyn={w_dyn}, w_query={w_query}"
        )
        self.w_dyn = w_dyn
        self.w_query = w_query
        self.asym_w = asym_w
        self.dynamism_window = dynamism_window
        assert cls_attn_method in self.VALID_CLS_METHODS, (
            f"cls_attn_method must be one of {self.VALID_CLS_METHODS}, "
            f"got '{cls_attn_method}'"
        )
        self.cls_attn_method = cls_attn_method

        # Top-K 软分配 + 垃圾桶
        self.topk_fusion = topk_fusion
        self.trash_ratio = trash_ratio

        # 融合方式
        assert fusion_method in self.VALID_FUSION_METHODS, (
            f"fusion_method must be one of {self.VALID_FUSION_METHODS}, "
            f"got '{fusion_method}'"
        )
        self.fusion_method = fusion_method

        # anchor-protected mean 权重
        assert 0.0 < anchor_weight <= 1.0, (
            f"anchor_weight must be in (0, 1], got {anchor_weight}"
        )
        self.anchor_weight = anchor_weight

        # 残留池 (v6)
        assert 0.0 <= residual_ratio < 1.0, (
            f"residual_ratio must be in [0, 1), got {residual_ratio}"
        )
        self.residual_ratio = residual_ratio
        self.res_imp_alpha = res_imp_alpha
        assert res_method in self.VALID_RES_METHODS, (
            f"res_method must be one of {self.VALID_RES_METHODS}, "
            f"got '{res_method}'"
        )
        self.res_method = res_method
        self.min_compression_ratio = min_compression_ratio

        # LLM 内部剪枝
        self.query_prune_layer = query_prune_layer
        self.llm_prune_ratio = llm_prune_ratio
        assert llm_prune_method in self.VALID_PRUNE_METHODS, (
            f"llm_prune_method must be one of {self.VALID_PRUNE_METHODS}, "
            f"got '{llm_prune_method}'"
        )
        self.llm_prune_method = llm_prune_method
        self.soft_prune_layer = soft_prune_layer

        # ViT 内部 token 融合
        self.vit_fusion_enabled = vit_fusion_enabled
        self.vit_fusion_group_layer = vit_fusion_group_layer
        self.vit_fusion_merge_layer = vit_fusion_merge_layer
        self.vit_fusion_sim_threshold = vit_fusion_sim_threshold
        self.vit_fusion_alpha = vit_fusion_alpha
        self.vit_fusion_fz_k = vit_fusion_fz_k
        assert vit_fusion_seg_method in self.VALID_VIT_FUSION_SEG_METHODS, (
            f"vit_fusion_seg_method must be one of {self.VALID_VIT_FUSION_SEG_METHODS}, "
            f"got '{vit_fusion_seg_method}'"
        )
        self.vit_fusion_seg_method = vit_fusion_seg_method
        assert vit_fusion_merge_mode in self.VALID_VIT_FUSION_MERGE_MODES, (
            f"vit_fusion_merge_mode must be one of {self.VALID_VIT_FUSION_MERGE_MODES}, "
            f"got '{vit_fusion_merge_mode}'"
        )
        self.vit_fusion_merge_mode = vit_fusion_merge_mode
        self.vit_fusion_temperature = vit_fusion_temperature

        # ---- 运行时状态 (由 pipeline 内部写入, 不由用户配置) ----
        self._visual_token_range = None
        self._target_budget = 0
        self._text_embeds = None
        self._s2_precomputed_token_norms = None
        self._s2_precomputed_cls_attn = None
        self._s2_precomputed_token_dyn = None
        self._s2_precomputed_token_uniq = None
        # ViT fusion 运行时状态
        self._vit_fusion_group_labels = None
        self._original_tokens_per_frame = None
        self._vit_fusion_per_frame_tokens = None
        self._vit_fusion_kept_indices_per_frame = None
        self._vit_fusion_valid_mask = None

    def __repr__(self):
        return (
            f"TensorDecompConfig(\n"
            f"  # ── Post-projector 压缩 ──\n"
            f"  stage2_retention_ratio={self.stage2_retention_ratio},\n"
            f"  dyseg_threshold={self.dyseg_threshold},\n"
            f"  cross_frame_lambda={self.cross_frame_lambda},\n"
            f"  fusion_temperature={self.fusion_temperature},\n"
            f"  min_segment_num={self.min_segment_num},\n"
            f"  complementary_segment={self.complementary_segment},\n"
            f"  # ── 组预算分配 ──\n"
            f"  group_budget_method='{self.group_budget_method}',\n"
            f"  effective_rank_k={self.effective_rank_k},\n"
            f"  # ── DPP 重要性信号 ──\n"
            f"  anchor_method='{self.anchor_method}',\n"
            f"  w_dyn={self.w_dyn},  # dynamism weight\n"
            f"  w_query={self.w_query},  # query score weight\n"
            f"  asym_w={self.asym_w},  # asymmetry in dynamism\n"
            f"  dynamism_window={self.dynamism_window},\n"
            f"  cls_attn_method='{self.cls_attn_method}',\n"
            f"  # ── Top-K 软分配 + 垃圾桶 ──\n"
            f"  topk_fusion={self.topk_fusion},\n"
            f"  trash_ratio={self.trash_ratio},\n"
            f"  fusion_method='{self.fusion_method}',\n"
            f"  anchor_weight={self.anchor_weight},\n"
            f"  residual_ratio={self.residual_ratio},\n"
            f"  res_imp_alpha={self.res_imp_alpha},\n"
            f"  res_method='{self.res_method}',\n"
            f"  min_compression_ratio={self.min_compression_ratio},\n"
            f"  # ── LLM 内部剪枝 ──\n"
            f"  query_prune_layer={self.query_prune_layer},\n"
            f"  llm_prune_ratio={self.llm_prune_ratio},\n"
            f"  llm_prune_method='{self.llm_prune_method}',\n"
            f"  soft_prune_layer={self.soft_prune_layer},\n"
            f"  # ── ViT 内部 token 融合 ──\n"
            f"  vit_fusion_enabled={self.vit_fusion_enabled},\n"
            f"  vit_fusion_group_layer={self.vit_fusion_group_layer},\n"
            f"  vit_fusion_merge_layer={self.vit_fusion_merge_layer},\n"
            f"  vit_fusion_sim_threshold={self.vit_fusion_sim_threshold},\n"
            f"  vit_fusion_alpha={self.vit_fusion_alpha},\n"
            f"  vit_fusion_fz_k={self.vit_fusion_fz_k},\n"
            f"  vit_fusion_seg_method='{self.vit_fusion_seg_method}',\n"
            f"  vit_fusion_merge_mode='{self.vit_fusion_merge_mode}',\n"
            f"  vit_fusion_temperature={self.vit_fusion_temperature},\n"
            f")"
        )