"""Static architectural descriptions of the study's models.

These are design facts (layers, parameter counts measured from the Stage-1 code at 384 px), never
performance numbers. Training status and metrics are always read from checkpoints / output files.
"""
from __future__ import annotations

from typing import Any, Dict

MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "model_b": {
        "name": "Hybrid Model B",
        "short": "Explanation-guided hybrid",
        "family": "hybrid",
        "role": "proposed",
        "architecture": "EfficientNet-B0 → 144 tokens → 2-layer Transformer → classification + localization heads",
        "description": ("The proposed model. Same network as Model A, but during training its spatial attention is "
                        "guided towards radiologist-annotated regions (RSNA boxes) so that it learns to base its "
                        "decision on the annotated lung opacity."),
        "training_objective": "classification + localization + attention guidance (λ > 0)",
        "parameters_millions": 5.2,
        "maps": ["localization", "attention", "gradcam"],
    },
    "model_a": {
        "name": "Hybrid Model A",
        "short": "Hybrid without guidance",
        "family": "hybrid",
        "role": "proposed (control)",
        "architecture": "EfficientNet-B0 → 144 tokens → 2-layer Transformer → classification + localization heads",
        "description": ("Identical architecture to Model B, trained to classify and localize but with no attention "
                        "guidance. Comparing A and B isolates the effect of explanation guidance."),
        "training_objective": "classification + localization",
        "parameters_millions": 5.2,
        "maps": ["localization", "attention", "gradcam"],
    },
    "resnet50": {
        "name": "ResNet50", "short": "CNN baseline", "family": "baseline", "role": "baseline",
        "architecture": "50-layer residual CNN (timm resnet50), single-logit head",
        "description": "Widely used convolutional baseline for chest X-ray classification.",
        "training_objective": "classification only", "parameters_millions": 23.5, "maps": ["gradcam"],
    },
    "densenet121": {
        "name": "DenseNet121", "short": "CNN baseline", "family": "baseline", "role": "baseline",
        "architecture": "121-layer densely connected CNN (timm densenet121), single-logit head",
        "description": "The backbone used by CheXNet-style chest X-ray classifiers.",
        "training_objective": "classification only", "parameters_millions": 7.0, "maps": ["gradcam"],
    },
    "efficientnet_b0": {
        "name": "EfficientNet-B0", "short": "CNN baseline", "family": "baseline", "role": "baseline",
        "architecture": "EfficientNet-B0 (timm), single-logit head",
        "description": ("Same backbone as the hybrid models without the Transformer and heads; shows what the "
                        "hybrid design adds."),
        "training_objective": "classification only", "parameters_millions": 4.0, "maps": ["gradcam"],
    },
    "vit_small": {
        "name": "ViT-Small", "short": "Transformer baseline", "family": "baseline", "role": "baseline",
        "architecture": "Vision Transformer, 16×16 patches at 384 px (timm vit_small_patch16_384)",
        "description": "Pure Transformer baseline without convolutional backbone.",
        "training_objective": "classification only", "parameters_millions": 21.8, "maps": ["gradcam"],
    },
    "ablation_cls_only": {
        "name": "Ablation: classification only", "short": "Ablation arm A", "family": "hybrid",
        "role": "ablation",
        "architecture": "Hybrid architecture; localization head present but untrained",
        "description": "Research control for the ablation study. Its localization map is not meaningful.",
        "training_objective": "classification only", "parameters_millions": 5.2, "maps": ["attention", "gradcam"],
    },
}

PRIMARY_MODELS = ["resnet50", "densenet121", "efficientnet_b0", "vit_small", "model_a", "model_b"]
TRAIN_MODULES = {
    "model_a": "ml.training.train_model_a", "model_b": "ml.training.train_model_b",
    "ablation_cls_only": "ml.training.train_ablation_cls_only", "resnet50": "ml.training.train_resnet50",
    "densenet121": "ml.training.train_densenet121", "efficientnet_b0": "ml.training.train_efficientnet",
    "vit_small": "ml.training.train_vit",
}
