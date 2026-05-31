

#!/bin/bash

# ============================================================================
# qwen3_ablation.sh  (DySeg-DPP — Qwen3-VL)
#
# 消融实验: 扫描核心参数 (DySeg + DPP + Top-K Soft Fusion)
# 适配 Qwen3-VL 架构 (DeepStack, 全局注意力 ViT, SigLIP-2, q_norm/k_norm)
#
# Pipeline:
#   ViT (全局注意力, 无 window) → Projector → 后投影信号计算 (CLS attn, Dynamism)
#   → DySeg 分组 → 组预算分配 (Effective Rank / MCD)
#   → 组内 DPP anchor 选择 → Top-K 软分配 + 自适应垃圾桶融合
#   → DeepStack 多层 ViT 特征注入 LLM (kept_global_indices 同步)
#   → LLM → (可选) 软剪枝 + 硬剪枝 (同步裁剪 DeepStack + visual_pos_masks)
#
# 可消融参数: 同 Qwen2.5-VL 版 ablation.sh (20 个参数)
#
# 基于 lmms-eval 框架，每次只变一个参数，其余固定为基线值
#
# Usage:
#   bash qwen3_ablation.sh
# ============================================================================

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=WARNING

# ############################################################################
# ██  模型配置
# ############################################################################

# TODO: 替换为你的 Qwen3-VL 模型路径
PRETRAINED="/mnt/bn/zhibo-demo/huiqiang/zhibo_demo/LLM_models/Qwen3-VL-8B-Instruct"
ATTN_IMPLEMENTATION="flash_attention_2"

# ############################################################################
# ██  视频处理
# ############################################################################

MAX_NUM_FRAMES=32
# Qwen3-VL 默认 pixel 配置 (SigLIP-2, patch_size=16, 32× spatial compression)
# MIN_PIXELS=50176
# MAX_PIXELS=200704

# ############################################################################
# ██  Baseline Values — DySeg-DPP 核心参数
# ############################################################################

# ── Post-projector 压缩 ──
BASE_STAGE2_RETENTION_RATIO=0.125
BASE_DYSEG_THRESHOLD=0.4
BASE_CROSS_FRAME_LAMBDA=0.0
BASE_FUSION_TEMPERATURE=0.01

# ── 组预算分配 ──
BASE_GROUP_BUDGET_METHOD=uniform
BASE_EFFECTIVE_RANK_K=32   # 只有 effective_rank2 才需要设置

# ── DPP 重要性信号 ──
BASE_W_DYN=0.5
BASE_ASYM_W=0.5
BASE_DYNAMISM_WINDOW=1
BASE_CLS_ATTN_METHOD=col_mean

# ── Top-K 软分配 + 垃圾桶 ──
BASE_TOPK_FUSION=10     # 8 10 12
BASE_FUSION_METHOD=mean
BASE_ANCHOR_WEIGHT=1

# 1 0.3   1 0.4    1 0.325
BASE_TRASH_RATIO=1.0 
BASE_RESIDUAL_RATIO=$(awk "BEGIN {printf \"%.4f\", $BASE_TRASH_RATIO / 3}")
# BASE_TRASH_RATIO=0.1 
# BASE_RESIDUAL_RATIO=0.0

# # 1 0.3   1 0.4    1 0.325
# BASE_TRASH_RATIO=0.5 
# BASE_RESIDUAL_RATIO=$(awk "BEGIN {printf \"%.4f\", $BASE_TRASH_RATIO / 5}")


# ── LLM 内部剪枝 ──
# Qwen3-VL-8B 有 36 层 decoder, 推荐剪枝层 28
BASE_QUERY_PRUNE_LAYER=28
BASE_LLM_PRUNE_RATIO=0.1
# BASE_LLM_PRUNE_RATIO=$(awk "BEGIN {printf \"%.4f\", (36 * 0.1 - ${BASE_STAGE2_RETENTION_RATIO:-0} * ${BASE_QUERY_PRUNE_LAYER:-0}) / (36 - ${BASE_QUERY_PRUNE_LAYER:-0}) / ${BASE_STAGE2_RETENTION_RATIO:-0} }")

BASE_LLM_PRUNE_METHOD=text_token    # text_token, last_token
BASE_SOFT_PRUNE_LAYER=-1


# ── DPP Anchor 选择方法 ──
BASE_ANCHOR_METHOD=facility_location     # facility_location  facility_location_optimized   dpp  dpp_optimized  dpc_knn

# ── DySeg 分段控制 ──
BASE_MIN_SEGMENT_NUM=8   # 8 16
BASE_COMPLEMENTARY_SEGMENT=True

# ── 残差池参数 ──
BASE_RES_IMP_ALPHA=0.3
BASE_RES_METHOD=dpc_knn


# ############################################################################
# ██  评测任务
# ############################################################################
# TASKS=("longvideobench_val_v" "videomme"  "egoschema" "mvbench" "mlvu_test" "mlvu_dev" "perceptiontest_val_mc_2000"  "perceptiontest_val_mc_5000")
# TASKS=( "videomme"  )
TASKS=(   "mvbench" "mlvu_test"  )

# ############################################################################
# ██  输出目录
# ############################################################################

OUTPUT_DIR="./eval_results/qwen3_ablation_study"
LOG_DIR="./eval_logs/qwen3_ablation_study"
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ============================================================================
# 通用运行函数
# ============================================================================
run_experiment() {
    local EXP_NAME=$1

    # 从 ABL_* 环境变量覆盖，未设则用 BASE_* 基线值
    local P_S2R=${ABL_STAGE2_RETENTION_RATIO:-$BASE_STAGE2_RETENTION_RATIO}
    local P_DST=${ABL_DYSEG_THRESHOLD:-$BASE_DYSEG_THRESHOLD}
    local P_CFL=${ABL_CROSS_FRAME_LAMBDA:-$BASE_CROSS_FRAME_LAMBDA}
    local P_FT=${ABL_FUSION_TEMPERATURE:-$BASE_FUSION_TEMPERATURE}
    local P_GBM=${ABL_GROUP_BUDGET_METHOD:-$BASE_GROUP_BUDGET_METHOD}
    local P_ERK=${ABL_EFFECTIVE_RANK_K:-$BASE_EFFECTIVE_RANK_K}
    local P_WDYN=${ABL_W_DYN:-$BASE_W_DYN}
    local P_AW=${ABL_ASYM_W:-$BASE_ASYM_W}
    local P_DWIN=${ABL_DYNAMISM_WINDOW:-$BASE_DYNAMISM_WINDOW}
    local P_CAM=${ABL_CLS_ATTN_METHOD:-$BASE_CLS_ATTN_METHOD}
    local P_TK=${ABL_TOPK_FUSION:-$BASE_TOPK_FUSION}
    local P_TR=${ABL_TRASH_RATIO:-$BASE_TRASH_RATIO}
    local P_FM=${ABL_FUSION_METHOD:-$BASE_FUSION_METHOD}
    local P_AWE=${ABL_ANCHOR_WEIGHT:-$BASE_ANCHOR_WEIGHT}
    local P_RR=${ABL_RESIDUAL_RATIO:-$BASE_RESIDUAL_RATIO}
    local P_QPL=${ABL_QUERY_PRUNE_LAYER:-$BASE_QUERY_PRUNE_LAYER}
    local P_LPR=${ABL_LLM_PRUNE_RATIO:-$BASE_LLM_PRUNE_RATIO}
    local P_LPM=${ABL_LLM_PRUNE_METHOD:-$BASE_LLM_PRUNE_METHOD}
    local P_SPL=${ABL_SOFT_PRUNE_LAYER:-$BASE_SOFT_PRUNE_LAYER}

    # ── 新增 9 个参数的 ABL_ 覆盖 ──
    local P_AM=${ABL_ANCHOR_METHOD:-$BASE_ANCHOR_METHOD}
    local P_MSN=${ABL_MIN_SEGMENT_NUM:-$BASE_MIN_SEGMENT_NUM}
    local P_CS=${ABL_COMPLEMENTARY_SEGMENT:-$BASE_COMPLEMENTARY_SEGMENT}
    local P_RIA=${ABL_RES_IMP_ALPHA:-$BASE_RES_IMP_ALPHA}
    local P_RM=${ABL_RES_METHOD:-$BASE_RES_METHOD}

    local RUN_DIR="${OUTPUT_DIR}/${TIMESTAMP}_${EXP_NAME}"
    mkdir -p ${RUN_DIR}

    cat > "${RUN_DIR}/params.json" << EOF
{
    "timestamp": "${TIMESTAMP}",
    "experiment": "${EXP_NAME}",
    "model": "Qwen3-VL",
    "params": {
        "stage2_retention_ratio": ${P_S2R},
        "dyseg_threshold": ${P_DST},
        "cross_frame_lambda": ${P_CFL},
        "fusion_temperature": ${P_FT},
        "group_budget_method": "${P_GBM}",
        "effective_rank_k": ${P_ERK},
        "w_dyn": ${P_WDYN},
        "asym_w": ${P_AW},
        "dynamism_window": ${P_DWIN},
        "cls_attn_method": "${P_CAM}",
        "topk_fusion": ${P_TK},
        "trash_ratio": ${P_TR},
        "fusion_method": "${P_FM}",
        "anchor_weight": ${P_AWE},
        "residual_ratio": ${P_RR},
        "query_prune_layer": ${P_QPL},
        "llm_prune_ratio": ${P_LPR},
        "llm_prune_method": "${P_LPM}",
        "soft_prune_layer": ${P_SPL},
        "anchor_method": "${P_AM}",
        "min_segment_num": ${P_MSN},
        "complementary_segment": ${P_CS},
        "res_imp_alpha": ${P_RIA},
        "res_method": "${P_RM}"
    }
}
EOF

    echo ""
    echo "============================================"
    echo " [Qwen3-VL] Ablation: ${EXP_NAME}"
    echo "============================================"
    echo "  stage2_retention_ratio:  ${P_S2R}"
    echo "  dyseg_threshold:         ${P_DST}"
    echo "  cross_frame_lambda:      ${P_CFL}"
    echo "  fusion_temperature:      ${P_FT}"
    echo "  group_budget_method:     ${P_GBM}"
    echo "  effective_rank_k:        ${P_ERK}"
    echo "  w_dyn:                   ${P_WDYN}"
    echo "  asym_w:                  ${P_AW}"
    echo "  dynamism_window:         ${P_DWIN}"
    echo "  cls_attn_method:         ${P_CAM}"
    echo "  topk_fusion:             ${P_TK}"
    echo "  trash_ratio:             ${P_TR}"
    echo "  fusion_method:           ${P_FM}"
    echo "  anchor_weight:           ${P_AWE}"
    echo "  residual_ratio:          ${P_RR}"
    echo "  query_prune_layer:       ${P_QPL}"
    echo "  llm_prune_ratio:         ${P_LPR}"
    echo "  llm_prune_method:        ${P_LPM}"
    echo "  soft_prune_layer:        ${P_SPL}"
    echo "  anchor_method:           ${P_AM}"
    echo "  min_segment_num:         ${P_MSN}"
    echo "  complementary_segment:   ${P_CS}"
    echo "  res_imp_alpha:           ${P_RIA}"
    echo "  res_method:              ${P_RM}"
    echo "  Output: ${RUN_DIR}"
    echo "============================================"

    for task in "${TASKS[@]}"; do
        echo "=========================================="
        echo "Evaluating task: $task"
        echo "=========================================="

        TASK_RUN_DIR="${RUN_DIR}/${task}"
        mkdir -p "${TASK_RUN_DIR}"

        EVAL_TASK="$task"
        LIMIT_ARGS=()
        if [ "$task" = "perceptiontest_val_mc_2000" ]; then
            LIMIT_ARGS=(--limit 2000)
            EVAL_TASK="perceptiontest_val_mc"
        fi
        if [ "$task" = "perceptiontest_val_mc_5000" ]; then
            LIMIT_ARGS=(--limit 5000)
            EVAL_TASK="perceptiontest_val_mc"
        fi

        accelerate launch --main_process_port 18888 \
            --num_processes 8 \
            -m lmms_eval \
            --model qwen3_vl \
            --model_args pretrained=${PRETRAINED},attn_implementation=${ATTN_IMPLEMENTATION},max_num_frames=${MAX_NUM_FRAMES},enable_tensor_decomp=True,stage2_retention_ratio=${P_S2R},dyseg_threshold=${P_DST},cross_frame_lambda=${P_CFL},fusion_temperature=${P_FT},group_budget_method=${P_GBM},effective_rank_k=${P_ERK},w_dyn=${P_WDYN},asym_w=${P_AW},dynamism_window=${P_DWIN},cls_attn_method=${P_CAM},topk_fusion=${P_TK},trash_ratio=${P_TR},fusion_method=${P_FM},anchor_weight=${P_AWE},residual_ratio=${P_RR},query_prune_layer=${P_QPL},llm_prune_ratio=${P_LPR},llm_prune_method=${P_LPM},soft_prune_layer=${P_SPL},anchor_method=${P_AM},min_segment_num=${P_MSN},complementary_segment=${P_CS},res_imp_alpha=${P_RIA},res_method=${P_RM} \
            --tasks ${EVAL_TASK} \
            --batch_size 1 \
            ${LIMIT_ARGS[@]} \
            --output_path "${TASK_RUN_DIR}" \
            --log_samples \
            --log_samples_suffix "qwen3_ablation_${EXP_NAME}" \
            2>&1 | tee "${LOG_DIR}/qwen3_ablation_${EXP_NAME}_${task}_${TIMESTAMP}.log"
        cp "${LOG_DIR}/qwen3_ablation_${EXP_NAME}_${task}_${TIMESTAMP}.log" "${TASK_RUN_DIR}/eval.log"

        # ── EgoSchema 自动提交 ──
        if [ "$task" = "egoschema" ]; then
            LATEST_SUBMISSION=$(find ${TASK_RUN_DIR} -name "inference_results_egoschema_MC_*.json" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2)
            if [ -n "$LATEST_SUBMISSION" ]; then
                SCORE_FILE="${LATEST_SUBMISSION%.json}_score.txt"
                echo "[EgoSchema] Auto-submitting: ${LATEST_SUBMISSION}"
                python tensor_decomp/validate.py --f "$LATEST_SUBMISSION" 2>&1 | tee "$SCORE_FILE"
            fi
        fi

        echo "  [DONE] ${EXP_NAME} / ${task} → ${TASK_RUN_DIR}"
    done
}

# # ============================================================================
# # Ablation 0: Baseline (使用全部默认参数运行一次)
# # ============================================================================
# echo ""
# echo "################################################################"
# echo "# Ablation 0: BASELINE (Qwen3-VL)"
# echo "################################################################"
# run_experiment "BASELINE"

# ============================================================================
# Ablation 1: STAGE2_RETENTION_RATIO 扫描
# ============================================================================
echo ""
echo "################################################################"
echo "# Ablation 1: STAGE2_RETENTION_RATIO"
echo "################################################################"
for S2R in  0.125  ; do
    ABL_STAGE2_RETENTION_RATIO=${S2R} run_experiment "S2R${S2R}"
done

# ============================================================================
# Ablation 2: W_DYN 扫描 (DPP importance dynamism weight)
# ============================================================================
# echo ""
# echo "################################################################"
# echo "# Ablation 2: W_DYN"
# echo "################################################################"
# for WD in 0.0 0.1 0.2 0.3 0.4 0.5; do
#     ABL_W_DYN=${WD} run_experiment "WDYN${WD}"
# done

# # ============================================================================
# # ============================================================================
# echo ""
# echo "################################################################"
# echo "################################################################"
# for WQ in   0.05 0.1 0.15 0.2; do
# done

# # ============================================================================
# # Ablation 4: ASYM_W 扫描 (dynamism asymmetry weight)
# # ============================================================================
# echo ""
# echo "################################################################"
# echo "# Ablation 4: ASYM_W"
# echo "################################################################"
# for AW in 0.0 1.0; do
#     ABL_ASYM_W=${AW} run_experiment "AW${AW}"
# done

# # ============================================================================
# # Ablation 5: TOPK_FUSION 扫描
# # ============================================================================
# echo ""
# echo "################################################################"
# echo "# Ablation 5: TOPK_FUSION"
# echo "################################################################"
# for TK in    1   ; do
#     ABL_TOPK_FUSION=${TK} run_experiment "TK${TK}"
# done

# # ============================================================================
# # Ablation 6: TRASH_RATIO 扫描
# # ============================================================================
# echo ""
# echo "################################################################"
# echo "# Ablation 6: TRASH_RATIO"
# echo "################################################################"
# for TR in     1.0 ; do
#     RR=$(awk "BEGIN{printf \"%.4f\", ${TR}/3}")
#     ABL_TRASH_RATIO=${TR} ABL_RESIDUAL_RATIO=${RR} run_experiment "TR${TR}"
# done

# # ============================================================================
# # Ablation 7: GROUP_BUDGET_METHOD 扫描
# # ============================================================================
# echo ""
# echo "################################################################"
# echo "# Ablation 7: GROUP_BUDGET_METHOD"
# echo "################################################################"
# for GBM in effective_rank  effective_rank2     ; do
#     ABL_GROUP_BUDGET_METHOD=${GBM} run_experiment "GBM_${GBM}"
# done

# # ============================================================================
# # Ablation 7.5: EFFECTIVE_RANK_K 扫描 (effective_rank2 randomized SVD q 系数)
# # ============================================================================
# echo ""
# echo "################################################################"
# echo "# Ablation 7.5: EFFECTIVE_RANK_K"
# echo "################################################################"
# for ERK in 32 16 8; do
#     ABL_GROUP_BUDGET_METHOD=effective_rank2 ABL_EFFECTIVE_RANK_K=${ERK} run_experiment "ERK${ERK}"
# done

# ============================================================================
# Ablation 8: FUSION_TEMPERATURE 扫描
# ============================================================================
# echo ""
# echo "################################################################"
# echo "# Ablation 8: FUSION_TEMPERATURE"
# echo "################################################################"
# for FT in 0.001 0.01 0.05 0.1 0.5; do
#     ABL_FUSION_TEMPERATURE=${FT} run_experiment "FT${FT}"
# done

# # ============================================================================
# # Ablation 9: DYSEG_THRESHOLD 扫描
# # ============================================================================
# echo ""
# echo "################################################################"
# echo "# Ablation 9: DYSEG_THRESHOLD"
# echo "################################################################"
# for DST in 0.5 0.55 0.6 0.65 0.70 0.75 0.80 0.85 ; do
#     ABL_DYSEG_THRESHOLD=${DST} run_experiment "DST${DST}"
# done

# ============================================================================
# Ablation 10: CLS_ATTN_METHOD 扫描
# ============================================================================
# echo ""
# echo "################################################################"
# echo "# Ablation 10: CLS_ATTN_METHOD"
# echo "################################################################"
# for CAM in pseudo_cls col_mean l2_norm adaptive col_mean_l2; do
#     ABL_CLS_ATTN_METHOD=${CAM} run_experiment "CAM_${CAM}"
# done

# # ============================================================================
# # Ablation 11: DYNAMISM_WINDOW 扫描
# # ============================================================================
# echo ""
# echo "################################################################"
# echo "# Ablation 11: DYNAMISM_WINDOW"
# echo "################################################################"
# for DWIN in   7; do
#     ABL_DYNAMISM_WINDOW=${DWIN} run_experiment "DWIN${DWIN}"
# done

# # ============================================================================
# # Ablation 12: CROSS_FRAME_LAMBDA 扫描
# # ============================================================================
# echo ""
# echo "################################################################"
# echo "# Ablation 12: CROSS_FRAME_LAMBDA"
# echo "################################################################"
# for CFL in 0.0 0.1 0.25 0.5 0.75 0.9 1.0; do
#     ABL_CROSS_FRAME_LAMBDA=${CFL} run_experiment "CFL${CFL}"
# done

# # ============================================================================
# # Ablation 13: FUSION_METHOD 扫描
# # ============================================================================
# echo ""
# echo "################################################################"
# echo "# Ablation 13: FUSION_METHOD"
# echo "################################################################"
# for FM in mean softmax softmax_imp; do
#     ABL_FUSION_METHOD=${FM} run_experiment "FM_${FM}"
# done

# # ============================================================================
# # Ablation 14: ANCHOR_WEIGHT 扫描 (仅 mean 方法)
# # ============================================================================
# echo ""
# echo "################################################################"
# echo "# Ablation 14: ANCHOR_WEIGHT (mean only)"
# echo "################################################################"
# for AW in 0.55 0.6 0.65 0.7 0.75 0.8; do
#     ABL_ANCHOR_WEIGHT=${AW} ABL_FUSION_METHOD=mean run_experiment "AW_${AW}"
# done

# # ============================================================================
# # Ablation 15: RESIDUAL_RATIO 扫描
# # ============================================================================
# echo ""
# echo "################################################################"
# echo "# Ablation 15: RESIDUAL_RATIO"
# echo "################################################################"
# for RR in  0.1  0.15 0.2 0.25  0.3 0.333 0.35  0.4 0.45 0.5  ; do
#     ABL_RESIDUAL_RATIO=${RR} run_experiment "RR${RR}"
# done

# # ============================================================================
# # Ablation 16: ANCHOR_METHOD 扫描
# # ============================================================================
# echo ""
# echo "################################################################"
# echo "# Ablation 16: ANCHOR_METHOD"
# echo "################################################################"
# for AM in facility_location_optimized ; do
#     ABL_ANCHOR_METHOD=${AM} run_experiment "AM_${AM}"
# done

# # ============================================================================
# # Ablation 17: MIN_SEGMENT_NUM 扫描
# # ============================================================================
# echo ""
# echo "################################################################"
# echo "# Ablation 17: MIN_SEGMENT_NUM"
# echo "################################################################"
# for MSN in   8   ; do
#     ABL_MIN_SEGMENT_NUM=${MSN} run_experiment "MSN${MSN}"
# done

# ============================================================================
# Ablation 18: RES_IMP_ALPHA 扫描
# ============================================================================
# echo ""
# echo "################################################################"
# echo "# Ablation 18: RES_IMP_ALPHA"
# echo "################################################################"
# for RIA in 0.1 0.2 0.3 0.5 0.7 1.0; do
#     ABL_RES_IMP_ALPHA=${RIA} run_experiment "RIA${RIA}"
# done

# ============================================================================
# Ablation 19: RES_METHOD 扫描
# ============================================================================
# echo ""
# echo "################################################################"
# echo "# Ablation 19: RES_METHOD"
# echo "################################################################"
# for RM in dpp dpc_knn; do
#     ABL_RES_METHOD=${RM} run_experiment "RM_${RM}"
# done

# ============================================================================
# Ablation 20: COMPLEMENTARY_SEGMENT 扫描
# ============================================================================
# echo ""
# echo "################################################################"
# echo "# Ablation 20: COMPLEMENTARY_SEGMENT"
# echo "################################################################"
# for CS in True False; do
#     ABL_COMPLEMENTARY_SEGMENT=${CS} run_experiment "CS_${CS}"
# done

# # ============================================================================
# # Ablation 21: QUERY_PRUNE_LAYER 扫描 (Qwen3-VL 特有: 28层 LLM)
# # ============================================================================
# echo ""
# echo "################################################################"
# echo "# Ablation 21: QUERY_PRUNE_LAYER (Qwen3-VL 28层)"
# echo "################################################################"
# for QPL in  20 22 24 26; do
#     ABL_QUERY_PRUNE_LAYER=${QPL} run_experiment "QPL${QPL}"
# done

# ============================================================================
# Ablation 22: LLM_PRUNE_RATIO 扫描
# ============================================================================
# echo ""
# echo "################################################################"
# echo "# Ablation 22: LLM_PRUNE_RATIO"
# echo "################################################################"
# for LPR in 0.1 0.2 0.3 0.4 0.5 1.0; do
#     ABL_LLM_PRUNE_RATIO=${LPR} run_experiment "LPR${LPR}"
# done


# # Ablation  soft layer
# for LP in last_token; do
#     ABL_LLM_PRUNE_METHOD=${LP} run_experiment "SL${LP}"
# done

# # Ablation  soft layer
# for SL in  "-1"   "16-17-18-19-20"; do
#     ABL_SOFT_PRUNE_LAYER=${SL} run_experiment "SL${SL}"
# done


# ============================================================================
# 汇总
# ============================================================================
echo ""
echo "================================================================"
echo " [Qwen3-VL] All ablation experiments complete!"
echo " Results directory: ${OUTPUT_DIR}"
echo " Logs directory:    ${LOG_DIR}"
echo "================================================================"
echo ""
echo " Ablation 0  - BASELINE:                (all default params)"
echo " Ablation 1  - STAGE2_RETENTION_RATIO:  0.05 / 0.10 / 0.125 / 0.15 / 0.20 / 0.30"
echo " Ablation 2  - W_DYN:                   0.0 / 0.1 / 0.2 / 0.3 / 0.4 / 0.5"
echo " Ablation 4  - ASYM_W:                  0.0 / 0.1 / 0.3 / 0.5 / 0.7 / 1.0"
echo " Ablation 5  - TOPK_FUSION:             1 / 2 / 3 / 4 / 5 / 8"
echo " Ablation 6  - TRASH_RATIO:             0.0 / 0.2 / 0.4 / 0.5 / 0.6 / 0.8 / 1.0"
echo " Ablation 7  - GROUP_BUDGET_METHOD:     effective_rank / mcd"
echo " Ablation 7.5 - EFFECTIVE_RANK_K:         16 / 32 / 64 / 96 / 128"
echo " Ablation 8  - FUSION_TEMPERATURE:      0.001 / 0.01 / 0.05 / 0.1 / 0.5"
echo " Ablation 9  - DYSEG_THRESHOLD:         0.70 / 0.80 / 0.85 / 0.90 / 0.95"
echo " Ablation 10 - CLS_ATTN_METHOD:         pseudo_cls / col_mean / l2_norm / adaptive / col_mean_l2"
echo " Ablation 11 - DYNAMISM_WINDOW:         1 / 2 / 3 / 5"
echo " Ablation 12 - CROSS_FRAME_LAMBDA:      0.0 / 0.1 / 0.25 / 0.5 / 1.0"
echo " Ablation 13 - FUSION_METHOD:           mean / softmax / softmax_imp"
echo " Ablation 14 - ANCHOR_WEIGHT (mean):    0.3 / 0.5 / 0.6 / 0.65 / 0.7 / 0.8 / 0.9"
echo " Ablation 15 - RESIDUAL_RATIO:          0.0 / 0.05 / 0.10 / 0.15 / 0.20 / 0.30"
echo " Ablation 16 - ANCHOR_METHOD:           dpp / leverage"
echo " Ablation 17 - MIN_SEGMENT_NUM:         0 / 2 / 4 / 8 / 16"
echo " Ablation 18 - RES_IMP_ALPHA:           0.1 / 0.2 / 0.3 / 0.5 / 0.7 / 1.0"
echo " Ablation 19 - RES_METHOD:              dpp / dpc_knn"
echo " Ablation 20 - COMPLEMENTARY_SEGMENT:   True / False"
echo " Ablation 21 - QUERY_PRUNE_LAYER:       -1 / 12 / 16 / 18 / 20 / 22 / 24  (Qwen3-VL 28层)"
echo " Ablation 22 - LLM_PRUNE_RATIO:         0.1 / 0.2 / 0.3 / 0.4 / 0.5 / 1.0"
echo "================================================================"







