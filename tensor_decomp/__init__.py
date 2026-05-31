"""
__init__.py

DySeg + DPP Anchor Selection + Top-K Soft Fusion 视频token压缩的注册入口。

Pipeline:
  ViT (vanilla) → Projector
    → DySeg 分组 (相邻帧按相似度阈值分组)
    → 组预算分配 (Effective Rank 或 MCD)
    → 组内 DPP anchor 选择 (importance = 凸组合 cls_attn + dynamism + query_score)
    → Top-K 软分配 + 自适应垃圾桶 Softmax 融合
    → LLM
    → (可选) 软剪枝 + 硬剪枝

Usage (Qwen2.5-VL):
    from tensor_decomp import tensor_decomp
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(...)
    tensor_decomp(model, stage2_retention_ratio=0.20, w_dyn=0.3, topk_fusion=3)

Usage (Qwen3-VL):
    from tensor_decomp import tensor_decomp_qwen3
    model = Qwen3VLForConditionalGeneration.from_pretrained(...)
    tensor_decomp_qwen3(model, stage2_retention_ratio=0.20, w_dyn=0.3, topk_fusion=3)

Usage (LLaVA-OneVision / LLaVA-Video):
    from tensor_decomp import tensor_decomp_llava
    model = load_pretrained_model(...)  # LlavaQwenForCausalLM
    tensor_decomp_llava(model, stage2_retention_ratio=0.20, w_dyn=0.3, topk_fusion=3)
"""

from .configuration_tensor_decomp import TensorDecompConfig

# ============================================================================
# Qwen2.5-VL imports
# ============================================================================
from .modeling_qwen2_5_vl import (
    Qwen2_5_VLAttention_forward,
    Qwen2_5_VLForConditionalGeneration_generate,
    Qwen2_5_VLModel_forward,
    Qwen2_5_VLModel_get_video_features,
    Qwen2_5_VLTextModel_forward,
    Qwen2_5_VLVisionAttention_forward,
    Qwen2_5_VLVisionBlock_forward,
    Qwen2_5_VisionTransformerPretrainedModel_forward,
)

# ============================================================================
# Qwen3-VL imports (lazy to avoid hard dependency on transformers with qwen3_vl)
# ============================================================================
_qwen3_vl_imported = False
_qwen3_vl_funcs = {}


def _ensure_qwen3_vl_imports():
    global _qwen3_vl_imported, _qwen3_vl_funcs
    if _qwen3_vl_imported:
        return
    from .modeling_qwen3_vl import (
        Qwen3VLVisionAttention_forward,
        Qwen3VLVisionBlock_forward,
        Qwen3VLVisionModel_forward,
        Qwen3VLModel_forward,
        Qwen3VLModel_get_video_features,
        Qwen3VLModel_get_image_features,
        Qwen3VLTextAttention_forward,
        Qwen3VLTextDecoderLayer_forward,
        Qwen3VLTextModel_forward,
        Qwen3VLForConditionalGeneration_generate,
    )
    _qwen3_vl_funcs.update(dict(
        Qwen3VLVisionAttention_forward=Qwen3VLVisionAttention_forward,
        Qwen3VLVisionBlock_forward=Qwen3VLVisionBlock_forward,
        Qwen3VLVisionModel_forward=Qwen3VLVisionModel_forward,
        Qwen3VLModel_forward=Qwen3VLModel_forward,
        Qwen3VLModel_get_video_features=Qwen3VLModel_get_video_features,
        Qwen3VLModel_get_image_features=Qwen3VLModel_get_image_features,
        Qwen3VLTextAttention_forward=Qwen3VLTextAttention_forward,
        Qwen3VLTextDecoderLayer_forward=Qwen3VLTextDecoderLayer_forward,
        Qwen3VLTextModel_forward=Qwen3VLTextModel_forward,
        Qwen3VLForConditionalGeneration_generate=Qwen3VLForConditionalGeneration_generate,
    ))
    _qwen3_vl_imported = True


# ============================================================================
# LLaVA-OneVision / LLaVA-Video imports (lazy)
# ============================================================================
_llava_imported = False
_llava_funcs = {}


def _ensure_llava_imports():
    global _llava_imported, _llava_funcs
    if _llava_imported:
        return
    from .modeling_llava_onevision import (
        SigLipAttention_forward,
        SigLipVisionTower_forward,
        LlavaMetaForCausalLM_encode_images,
        LlavaMetaForCausalLM_prepare_inputs_labels_for_multimodal,
        Qwen2Attention_forward,
        Qwen2DecoderLayer_forward,
        Qwen2Model_forward,
    )
    _llava_funcs.update(dict(
        SigLipAttention_forward=SigLipAttention_forward,
        SigLipVisionTower_forward=SigLipVisionTower_forward,
        LlavaMetaForCausalLM_encode_images=LlavaMetaForCausalLM_encode_images,
        LlavaMetaForCausalLM_prepare_inputs_labels_for_multimodal=LlavaMetaForCausalLM_prepare_inputs_labels_for_multimodal,
        Qwen2Attention_forward=Qwen2Attention_forward,
        Qwen2DecoderLayer_forward=Qwen2DecoderLayer_forward,
        Qwen2Model_forward=Qwen2Model_forward,
    ))
    _llava_imported = True


# ============================================================================
# Shared parameter signature for both tensor_decomp and tensor_decomp_qwen3
# ============================================================================
_TD_PARAM_DEFAULTS = dict(
    stage2_retention_ratio=0.20,
    dyseg_threshold=0.85,
    cross_frame_lambda=0.0,
    fusion_temperature=0.01,
    min_segment_num=0,
    complementary_segment=True,
    group_budget_method="effective_rank",
    effective_rank_k=64,
    anchor_method="dpp",
    w_dyn=0.3,
    w_query=0.05,
    asym_w=0.3,
    dynamism_window=1,
    cls_attn_method="pseudo_cls",
    topk_fusion=3,
    trash_ratio=0.6,
    fusion_method="mean",
    anchor_weight=0.5,
    residual_ratio=0.0,
    res_imp_alpha=0.3,
    res_method="dpc_knn",
    query_prune_layer=21,
    llm_prune_ratio=1.0,
    llm_prune_method="text_token",
    soft_prune_layer=-1,
)


def _build_td_config(**kwargs):
    """Build TensorDecompConfig from kwargs, filling defaults from _TD_PARAM_DEFAULTS."""
    params = {k: kwargs.get(k, v) for k, v in _TD_PARAM_DEFAULTS.items()}
    return TensorDecompConfig(**params)


# ============================================================================
# Qwen2.5-VL entry point (unchanged)
# ============================================================================
def tensor_decomp(
    model,
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
    # ---- LLM 内部剪枝 ----
    query_prune_layer: int = 21,
    llm_prune_ratio: float = 1.0,
    llm_prune_method: str = "text_token",
    soft_prune_layer: int = -1,
):
    """在 Qwen2.5-VL 模型上注册 DySeg + DPP + Top-K Soft Fusion 压缩。

    此函数执行以下操作:
    1. 创建 TensorDecompConfig
    2. Monkey-patch 全部 8 个函数
    3. 将 config 通过 setattr 挂载到 model 各组件上

    Returns:
        model: 注册后的模型。
    """
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLAttention,
        Qwen2_5_VLForConditionalGeneration,
        Qwen2_5_VLModel,
        Qwen2_5_VLTextModel,
        Qwen2_5_VLVisionAttention,
        Qwen2_5_VLVisionBlock,
        Qwen2_5_VisionTransformerPretrainedModel,
    )

    assert isinstance(model, Qwen2_5_VLForConditionalGeneration), (
        f"Expected Qwen2_5_VLForConditionalGeneration, got {type(model)}"
    )

    td_config = _build_td_config(**locals())

    # Monkey-patch 全部 8 个函数
    Qwen2_5_VLAttention.forward = Qwen2_5_VLAttention_forward
    Qwen2_5_VLModel.get_video_features = Qwen2_5_VLModel_get_video_features
    Qwen2_5_VLTextModel.forward = Qwen2_5_VLTextModel_forward
    Qwen2_5_VLModel.forward = Qwen2_5_VLModel_forward
    Qwen2_5_VLVisionBlock.forward = Qwen2_5_VLVisionBlock_forward
    Qwen2_5_VLVisionAttention.forward = Qwen2_5_VLVisionAttention_forward
    Qwen2_5_VisionTransformerPretrainedModel.forward = (
        Qwen2_5_VisionTransformerPretrainedModel_forward
    )

    # 保存原始 generate 并替换
    Qwen2_5_VLForConditionalGeneration.generate_ori = (
        Qwen2_5_VLForConditionalGeneration.generate
    )
    Qwen2_5_VLForConditionalGeneration.generate = (
        Qwen2_5_VLForConditionalGeneration_generate
    )

    # 将 config 挂载到所有相关组件
    setattr(model, 'td_config', td_config)
    setattr(model.model, 'td_config', td_config)
    setattr(model.model.language_model, 'td_config', td_config)
    setattr(model.model.visual, 'td_config', td_config)

    print(f"[TensorDecomp] Registered DySeg + DPP + Top-K Soft Fusion compression (Qwen2.5-VL):")
    print(f"  {td_config}")

    return model


# ============================================================================
# Qwen3-VL entry point (NEW)
# ============================================================================
def tensor_decomp_qwen3(
    model,
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
    # ---- LLM 内部剪枝 ----
    query_prune_layer: int = 21,
    llm_prune_ratio: float = 1.0,
    llm_prune_method: str = "text_token",
    soft_prune_layer: int = -1,
):
    """在 Qwen3-VL 模型上注册 DySeg + DPP + Top-K Soft Fusion 压缩。

    与 tensor_decomp() 功能完全相同, 但适配 Qwen3-VL 架构:
    - DeepStack (多层 ViT 特征注入 LLM)
    - 无 window attention (全局注意力)
    - 无 sliding window LLM
    - SigLIP-2 ViT (patch_size=16, 32× spatial compression)
    - q_norm/k_norm in LLM attention

    此函数执行以下操作:
    1. 创建 TensorDecompConfig
    2. Monkey-patch 全部 10 个函数 (比 Qwen2.5-VL 多 DeepStack/image 相关)
    3. 将 config 通过 setattr 挂载到 model 各组件上

    Returns:
        model: 注册后的模型。
    """
    _ensure_qwen3_vl_imports()

    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLForConditionalGeneration,
        Qwen3VLVisionAttention,
        Qwen3VLVisionBlock,
        Qwen3VLVisionModel,
        Qwen3VLModel,
        Qwen3VLTextAttention,
        Qwen3VLTextDecoderLayer,
        Qwen3VLTextModel,
    )

    assert isinstance(model, Qwen3VLForConditionalGeneration), (
        f"Expected Qwen3VLForConditionalGeneration, got {type(model)}"
    )

    td_config = _build_td_config(**locals())

    f = _qwen3_vl_funcs  # shorthand

    # Monkey-patch 全部 10 个函数
    Qwen3VLVisionAttention.forward = f['Qwen3VLVisionAttention_forward']
    Qwen3VLVisionBlock.forward = f['Qwen3VLVisionBlock_forward']
    Qwen3VLVisionModel.forward = f['Qwen3VLVisionModel_forward']
    Qwen3VLModel.forward = f['Qwen3VLModel_forward']
    Qwen3VLModel.get_video_features = f['Qwen3VLModel_get_video_features']
    Qwen3VLModel.get_image_features = f['Qwen3VLModel_get_image_features']
    Qwen3VLTextAttention.forward = f['Qwen3VLTextAttention_forward']
    Qwen3VLTextDecoderLayer.forward = f['Qwen3VLTextDecoderLayer_forward']
    Qwen3VLTextModel.forward = f['Qwen3VLTextModel_forward']

    # 保存原始 generate 并替换
    Qwen3VLForConditionalGeneration.generate_ori = (
        Qwen3VLForConditionalGeneration.generate
    )
    Qwen3VLForConditionalGeneration.generate = (
        f['Qwen3VLForConditionalGeneration_generate']
    )

    # 将 config 挂载到所有相关组件
    setattr(model, 'td_config', td_config)
    setattr(model.model, 'td_config', td_config)
    setattr(model.model.language_model, 'td_config', td_config)
    setattr(model.model.visual, 'td_config', td_config)

    print(f"[TensorDecomp] Registered DySeg + DPP + Top-K Soft Fusion compression (Qwen3-VL):")
    print(f"  {td_config}")

    return model


# ============================================================================
# LLaVA-OneVision / LLaVA-Video entry point
# ============================================================================
def tensor_decomp_llava(
    model,
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
    # ---- LLM 内部剪枝 ----
    query_prune_layer: int = 21,
    llm_prune_ratio: float = 1.0,
    llm_prune_method: str = "text_token",
    soft_prune_layer: int = -1,
):
    """在 LLaVA-OneVision / LLaVA-Video 模型上注册 DySeg + DPP + Top-K Soft Fusion 压缩。

    适配 LLaVA 架构 (LlavaQwenForCausalLM + SigLIP ViT + Qwen2 LLM):
    - SigLIP vision encoder (无内部 token 融合, 因此没有 vit_fusion_* 参数)
    - Qwen2 LLM backbone
    - LlavaMetaForCausalLM 负责 encode_images 和 prepare_inputs_labels_for_multimodal

    此函数执行以下操作:
    1. 创建 TensorDecompConfig
    2. Monkey-patch 全部 7 个函数
    3. 将 config 通过 setattr 挂载到 model 各组件上

    Note:
        No ViT fusion parameters since SigLIP doesn't support internal token fusion.

    Returns:
        model: 注册后的模型。
    """
    _ensure_llava_imports()

    try:
        from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
    except ImportError as e:
        raise ImportError(
            "LLaVA is not installed. Please install llava to use tensor_decomp_llava()."
        ) from e

    try:
        from llava.model.llava_arch import LlavaMetaForCausalLM
    except ImportError as e:
        raise ImportError(
            "Cannot import LlavaMetaForCausalLM from llava.model.llava_arch."
        ) from e

    from llava.model.multimodal_encoder.siglip_encoder import SigLipAttention, SigLipVisionTower
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2Attention,
        Qwen2DecoderLayer,
        Qwen2Model,
    )

    assert isinstance(model, LlavaQwenForCausalLM), (
        f"Expected LlavaQwenForCausalLM, got {type(model)}"
    )

    td_config = _build_td_config(**{k: v for k, v in locals().items() if k != 'model'})

    f = _llava_funcs  # shorthand

    # Monkey-patch 全部 7 个函数
    SigLipAttention.forward = f['SigLipAttention_forward']
    # For SigLipVisionTower, get the class from the model instance
    type(model.model.vision_tower).forward = f['SigLipVisionTower_forward']
    LlavaMetaForCausalLM.encode_images = f['LlavaMetaForCausalLM_encode_images']
    LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = f['LlavaMetaForCausalLM_prepare_inputs_labels_for_multimodal']
    Qwen2Attention.forward = f['Qwen2Attention_forward']
    Qwen2DecoderLayer.forward = f['Qwen2DecoderLayer_forward']
    Qwen2Model.forward = f['Qwen2Model_forward']

    # 将 config 挂载到所有相关组件
    setattr(model, 'td_config', td_config)
    setattr(model.model, 'td_config', td_config)

    print(f"[TensorDecomp] Registered DySeg + DPP + Top-K Soft Fusion compression (LLaVA-OneVision / LLaVA-Video):")
    print(f"  {td_config}")

    return model