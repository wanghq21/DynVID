#!/bin/bash

# ============================================================================
# eval_tensor_decomp.sh  (DySeg-DPP — DPP Anchor + Top-K Soft Fusion)
#
# DySeg + DPP Anchor Selection + Top-K Soft Fusion 视频 token 压缩的评测脚本
# 基于 lmms-eval 框架
#
# Pipeline:
#   ViT (完全不动) → Projector → 后投影信号计算 (CLS attn, Dynamism)
#   → DySeg 分组 (相邻帧按余弦相似度阈值分组)
#   → 组预算分配 (Effective Rank 或 MCD)
#   → 组内 DPP anchor 选择 (importance = 凸组合 cls_attn + dynamism + query)
#   → Top-K 软分配 + 自适应垃圾桶 Softmax 融合
#   → LLM
#   → (可选) 软剪枝 + 硬剪枝 (attention-score-only, 物理移除 token)
#
# Usage:
#   bash eval_tensor_decomp.sh
# ============================================================================

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=WARNING


# ############################################################################
# ██  模型配置 (Model Config)
# ############################################################################

PRETRAINED="/mnt/bn/zhibo-demo/huiqiang/zhibo_demo/LLM_models/Qwen2.5-VL-7B-Instruct"
ATTN_IMPLEMENTATION="flash_attention_2"


# ############################################################################
# ██  视频处理 (Video Processing)
# ############################################################################

MAX_NUM_FRAMES=32
MIN_PIXELS=50176
MAX_PIXELS=200704


# ############################################################################
# ██  DySeg-DPP 核心参数
# ############################################################################

# ── 总压缩预算 (Stage 2 Retention Ratio) ──
# budget = T × N × ratio, 越小压缩越激进
STAGE2_RETENTION_RATIO=0.20

# ── DySeg 分组阈值 ──
# 相邻帧 cosine similarity >= threshold → 同组
DYSEG_THRESHOLD=0.85

# ── 跨帧时间距离惩罚 ──
# sim -= λ × |frame_i - frame_j|, 0=无惩罚, inf=帧内硬约束
CROSS_FRAME_LAMBDA=0.0

# ── Softmax 融合温度 ──
# 越小越锐利 (接近 hard assignment), 越大越平滑
FUSION_TEMPERATURE=0.01

# ── 组预算分配方法 ──
# "effective_rank": SVD有效秩 (推荐, 已包含组大小信息)
# "mcd": Mean Cosine Distance × len(group)
GROUP_BUDGET_METHOD=effective_rank

# ── DPP 重要性: dynamism 权重 ──
# importance = (1-w_dyn-w_query)*cls + w_dyn*dyn + w_query*query
# 注意: w_dyn + w_query < 1.0
W_DYN=0.3

# ── DPP 重要性: query score 权重 ──
W_QUERY=0.05

# ── Dynamism 中 asymmetry 权重 ──
# dynamism = (1-asym_w)*novelty + asym_w*|fwd_max - bwd_max|
# novelty = 1 - max(fwd_max, bwd_max)
ASYM_W=0.3

# ── 动态性时间窗口 ──
DYNAMISM_WINDOW=1

# ── CLS Attention 方法 ──
# pseudo_cls / col_mean / l2_norm / adaptive
CLS_ATTN_METHOD=pseudo_cls

# ── Top-K 软分配: 每个 drop token 分配到的 anchor 数量 ──
# 1 = 退化为硬分配, 推荐 3~5
TOPK_FUSION=3

# ── 自适应垃圾桶: 阈值比例 ──
# threshold = per_token_max_sim_among_drops × ratio
# 冗余 drop → 高 threshold → 容易丢弃; 独特 drop → 低 threshold → 保留
# 0 = 无垃圾桶, 推荐 0.4~0.8
TRASH_RATIO=0.6

# ── 融合方式 ──
# "mean": 纯矩阵乘均值融合 (最快)
# "softmax": scatter softmax 融合
# "softmax_imp": softmax + importance 加权
FUSION_METHOD=mean

# ── mean 融合 anchor 权重 α ──
# fused = α * anchor + (1-α) * drop_centroid
# α=0.5 → anchor 贡献 50%, α=1.0 → 不融合 (仅 mean 方法有效)
ANCHOR_WEIGHT=0.5

# ── 残留池预留预算比例 (v6) ──
# K_anchor = round(K_group × (1-ratio)), K_residual = K_group - K_anchor
# 0 = 禁用残留池 (v5兼容), 推荐 0.10~0.20
RESIDUAL_RATIO=0.0

# ── LLM 剪枝层号 (-1 = 禁用) ──
QUERY_PRUNE_LAYER=21

# ── LLM 剪枝保留比 (1.0=不剪) ──
LLM_PRUNE_RATIO=1.0

# ── LLM 剪枝注意力评分方法 ──
LLM_PRUNE_METHOD=text_token

# ── 软剪枝层号 (-1 = 禁用) ──
SOFT_PRUNE_LAYER=-1

# ── ViT 内部 token 融合 ──
VIT_FUSION_ENABLED=False
VIT_FUSION_GROUP_LAYER=15
VIT_FUSION_MERGE_LAYER=25
VIT_FUSION_SIM_THRESHOLD=0.85
VIT_FUSION_ALPHA=0.5

# ── DPP anchor 选择方法 ──
# "dpp": 标准 DPP 行列式点过程 (默认)
# "leverage": leverage score 采样
ANCHOR_METHOD=dpp

# ── DySeg 分段控制 ──
# min_segment_num: 最小分段数, 0=不限制
MIN_SEGMENT_NUM=0

# complementary_segment: 是否启用互补分段
COMPLEMENTARY_SEGMENT=True

# ── 残差池参数 ──
# res_imp_alpha: DPP₂ importance 压缩指数
RES_IMP_ALPHA=0.3

# res_method: 残差池处理方法 "dpp" / "dpc_knn"
RES_METHOD=dpc_knn

# ── ViT 内部 token 融合 (高级参数) ──
# vit_fusion_fz_k: farthest zone k, -1.0=禁用
VIT_FUSION_FZ_K=-1.0

# vit_fusion_seg_method: ViT fusion 分组方法
VIT_FUSION_SEG_METHOD=threshold_cc

# vit_fusion_merge_mode: ViT fusion 合并模式
VIT_FUSION_MERGE_MODE=mean

# vit_fusion_temperature: ViT fusion softmax 温度
VIT_FUSION_TEMPERATURE=0.1


# ############################################################################
# ██  评测任务
# ############################################################################

TASKS="mvbench"


# ############################################################################
# ██  输出目录与运行配置
# ############################################################################

OUTPUT_DIR="./eval_results/tensor_decomp"
LOG_DIR="./eval_logs/tensor_decomp"
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="${TIMESTAMP}_S2R${STAGE2_RETENTION_RATIO}_DST${DYSEG_THRESHOLD}_DPP_wdyn${W_DYN}_topk${TOPK_FUSION}"
RUN_OUTPUT_DIR="${OUTPUT_DIR}/${RUN_NAME}"
mkdir -p ${RUN_OUTPUT_DIR}

# ---- 保存参数到 JSON ----
cat > "${RUN_OUTPUT_DIR}/params.json" << EOF
{
    "timestamp": "${TIMESTAMP}",
    "model": {
        "pretrained": "${PRETRAINED}",
        "attn_implementation": "${ATTN_IMPLEMENTATION}"
    },
    "video": {
        "max_num_frames": ${MAX_NUM_FRAMES},
        "min_pixels": ${MIN_PIXELS},
        "max_pixels": ${MAX_PIXELS}
    },
    "dyseg_dpp_params": {
        "stage2_retention_ratio": ${STAGE2_RETENTION_RATIO},
        "dyseg_threshold": ${DYSEG_THRESHOLD},
        "cross_frame_lambda": ${CROSS_FRAME_LAMBDA},
        "fusion_temperature": ${FUSION_TEMPERATURE},
        "group_budget_method": "${GROUP_BUDGET_METHOD}",
        "w_dyn": ${W_DYN},
        "w_query": ${W_QUERY},
        "asym_w": ${ASYM_W},
        "dynamism_window": ${DYNAMISM_WINDOW},
        "cls_attn_method": "${CLS_ATTN_METHOD}",
        "topk_fusion": ${TOPK_FUSION},
        "trash_ratio": ${TRASH_RATIO},
        "fusion_method": "${FUSION_METHOD}",
        "anchor_weight": ${ANCHOR_WEIGHT},
        "residual_ratio": ${RESIDUAL_RATIO},
        "query_prune_layer": ${QUERY_PRUNE_LAYER},
        "llm_prune_ratio": ${LLM_PRUNE_RATIO},
        "llm_prune_method": "${LLM_PRUNE_METHOD}",
        "soft_prune_layer": ${SOFT_PRUNE_LAYER},
        "vit_fusion_enabled": ${VIT_FUSION_ENABLED},
        "vit_fusion_group_layer": ${VIT_FUSION_GROUP_LAYER},
        "vit_fusion_merge_layer": ${VIT_FUSION_MERGE_LAYER},
        "vit_fusion_sim_threshold": ${VIT_FUSION_SIM_THRESHOLD},
        "vit_fusion_alpha": ${VIT_FUSION_ALPHA},
        "anchor_method": "${ANCHOR_METHOD}",
        "min_segment_num": ${MIN_SEGMENT_NUM},
        "complementary_segment": ${COMPLEMENTARY_SEGMENT},
        "res_imp_alpha": ${RES_IMP_ALPHA},
        "res_method": "${RES_METHOD}",
        "vit_fusion_fz_k": ${VIT_FUSION_FZ_K},
        "vit_fusion_seg_method": "${VIT_FUSION_SEG_METHOD}",
        "vit_fusion_merge_mode": "${VIT_FUSION_MERGE_MODE}",
        "vit_fusion_temperature": ${VIT_FUSION_TEMPERATURE}
    },
    "eval": {
        "tasks": "${TASKS}",
        "batch_size": 1,
        "num_processes": 8
    },
    "run_name": "${RUN_NAME}"
}
EOF

echo "Parameters saved to: ${RUN_OUTPUT_DIR}/params.json"

# ---- 打印配置 ----
echo "============================================"
echo "DySeg-DPP Video Token Compression"
echo "============================================"
echo ""
echo "── 模型配置 ──"
echo "Model:                  ${PRETRAINED}"
echo "attn_impl:              ${ATTN_IMPLEMENTATION}"
echo ""
echo "── 视频处理 ──"
echo "Max frames:             ${MAX_NUM_FRAMES}"
echo "Min/Max pixels:         ${MIN_PIXELS} / ${MAX_PIXELS}"
echo ""
echo "── DySeg-DPP 核心参数 ──"
echo "stage2_retention_ratio: ${STAGE2_RETENTION_RATIO}"
echo "dyseg_threshold:        ${DYSEG_THRESHOLD}"
echo "cross_frame_lambda:     ${CROSS_FRAME_LAMBDA}"
echo "fusion_temperature:     ${FUSION_TEMPERATURE}"
echo "group_budget_method:    ${GROUP_BUDGET_METHOD}"
echo "w_dyn:                  ${W_DYN}  (DPP importance dynamism weight)"
echo "w_query:                ${W_QUERY}  (DPP importance query weight)"
echo "asym_w:                 ${ASYM_W}  (dynamism asymmetry weight)"
echo "dynamism_window:        ${DYNAMISM_WINDOW}"
echo "cls_attn_method:        ${CLS_ATTN_METHOD}"
echo "topk_fusion:            ${TOPK_FUSION}  (Top-K soft assignment)"
echo "trash_ratio:            ${TRASH_RATIO}  (adaptive trash bin ratio)"
echo "fusion_method:          ${FUSION_METHOD}  (mean / softmax / softmax_imp)"
echo "anchor_weight:          ${ANCHOR_WEIGHT}  (mean融合anchor权重α)"
echo "residual_ratio:         ${RESIDUAL_RATIO}  (残留池预留比例, 0=禁用)"
echo "query_prune_layer:      ${QUERY_PRUNE_LAYER}"
echo "llm_prune_ratio:        ${LLM_PRUNE_RATIO}"
echo "llm_prune_method:       ${LLM_PRUNE_METHOD}"
echo "soft_prune_layer:       ${SOFT_PRUNE_LAYER}"
echo ""
echo "── DPP Anchor 选择 ──"
echo "anchor_method:          ${ANCHOR_METHOD}  (dpp / leverage)"
echo "min_segment_num:        ${MIN_SEGMENT_NUM}  (最小分段数, 0=不限制)"
echo "complementary_segment:  ${COMPLEMENTARY_SEGMENT}"
echo ""
echo "── 残差池参数 ──"
echo "res_imp_alpha:          ${RES_IMP_ALPHA}  (DPP₂ importance 压缩指数)"
echo "res_method:             ${RES_METHOD}  (残差池处理: dpp / dpc_knn)"
echo ""
echo "── ViT 内部 token 融合 ──"
echo "vit_fusion_enabled:     ${VIT_FUSION_ENABLED}"
echo "vit_fusion_fz_k:        ${VIT_FUSION_FZ_K}  (farthest zone k, -1=禁用)"
echo "vit_fusion_seg_method:  ${VIT_FUSION_SEG_METHOD}"
echo "vit_fusion_merge_mode:  ${VIT_FUSION_MERGE_MODE}"
echo "vit_fusion_temperature: ${VIT_FUSION_TEMPERATURE}"
echo ""
echo "── 评测任务 ──"
echo "Tasks:                  ${TASKS}"
echo "Output:                 ${RUN_OUTPUT_DIR}"
echo "============================================"

# ---- 运行评测 ----
accelerate launch --main_process_port 18888 \
    --num_processes 8 \
    -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=${PRETRAINED},attn_implementation=${ATTN_IMPLEMENTATION},max_num_frames=${MAX_NUM_FRAMES},min_pixels=${MIN_PIXELS},max_pixels=${MAX_PIXELS},enable_tensor_decomp=True,stage2_retention_ratio=${STAGE2_RETENTION_RATIO},dyseg_threshold=${DYSEG_THRESHOLD},cross_frame_lambda=${CROSS_FRAME_LAMBDA},fusion_temperature=${FUSION_TEMPERATURE},group_budget_method=${GROUP_BUDGET_METHOD},w_dyn=${W_DYN},w_query=${W_QUERY},asym_w=${ASYM_W},dynamism_window=${DYNAMISM_WINDOW},cls_attn_method=${CLS_ATTN_METHOD},topk_fusion=${TOPK_FUSION},trash_ratio=${TRASH_RATIO},fusion_method=${FUSION_METHOD},anchor_weight=${ANCHOR_WEIGHT},residual_ratio=${RESIDUAL_RATIO},query_prune_layer=${QUERY_PRUNE_LAYER},llm_prune_ratio=${LLM_PRUNE_RATIO},llm_prune_method=${LLM_PRUNE_METHOD},soft_prune_layer=${SOFT_PRUNE_LAYER},vit_fusion_enabled=${VIT_FUSION_ENABLED},vit_fusion_group_layer=${VIT_FUSION_GROUP_LAYER},vit_fusion_merge_layer=${VIT_FUSION_MERGE_LAYER},vit_fusion_sim_threshold=${VIT_FUSION_SIM_THRESHOLD},vit_fusion_alpha=${VIT_FUSION_ALPHA},anchor_method=${ANCHOR_METHOD},min_segment_num=${MIN_SEGMENT_NUM},complementary_segment=${COMPLEMENTARY_SEGMENT},res_imp_alpha=${RES_IMP_ALPHA},res_method=${RES_METHOD},vit_fusion_fz_k=${VIT_FUSION_FZ_K},vit_fusion_seg_method=${VIT_FUSION_SEG_METHOD},vit_fusion_merge_mode=${VIT_FUSION_MERGE_MODE},vit_fusion_temperature=${VIT_FUSION_TEMPERATURE} \
    --tasks ${TASKS} \
    --batch_size 1 \
    --output_path "${RUN_OUTPUT_DIR}" \
    --log_samples \
    --log_samples_suffix "dyseg_dpp_S2R${STAGE2_RETENTION_RATIO}_DST${DYSEG_THRESHOLD}" \
    2>&1 | tee "${LOG_DIR}/eval_${TIMESTAMP}.log"

cp "${LOG_DIR}/eval_${TIMESTAMP}.log" "${RUN_OUTPUT_DIR}/eval.log"

echo ""
echo "Evaluation complete."
echo "  Results:    ${RUN_OUTPUT_DIR}"
echo "  Parameters: ${RUN_OUTPUT_DIR}/params.json"
echo "  Log:        ${RUN_OUTPUT_DIR}/eval.log"

# ============ EgoSchema 自动提交评测 ============
if echo "${TASKS}" | grep -qw "egoschema"; then
    LATEST_SUBMISSION=$(find ${OUTPUT_DIR} -name "inference_results_egoschema_MC_*.json" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2)
    if [ -n "$LATEST_SUBMISSION" ]; then
        echo ""
        echo "[EgoSchema] Auto-submitting: ${LATEST_SUBMISSION}"
        SCORE_FILE="${LATEST_SUBMISSION%.json}_score.txt"
        python tensor_decomp/validate.py --f "$LATEST_SUBMISSION" 2>&1 | tee "$SCORE_FILE"
        echo "[EgoSchema] Score saved to: ${SCORE_FILE}"
    else
        echo "[EgoSchema] No submission file found in ${OUTPUT_DIR}"
    fi
fi
