import numpy as np
import pytest
import torch

from ml.explainability.gradcam_pp import GradCAMPlusPlus, get_module, normalize_maps
from ml.explainability.visualization import (comparison_panel, draw_boxes, encode_png_base64, normalize_heatmap,
                                             overlay_heatmap)
from ml.models.factory import build_model
from ml.utils.config import load_config


@pytest.mark.parametrize("name", ["hybrid", "resnet50", "densenet121", "efficientnet_b0", "vit_small"])
def test_gradcam_pp_all_models(name):
    torch.manual_seed(0)
    cfg = dict(load_config("hybrid_common.yaml")["model"]) if name == "hybrid" else {"name": name}
    cfg["pretrained_backbone"] = False
    m = build_model(cfg, image_size=64)
    maps = GradCAMPlusPlus(m)(torch.randn(2, 3, 64, 64))
    assert maps.shape == (2, 64, 64)
    assert maps.min() >= 0 and maps.max() <= 1
    assert m.training  # original mode restored
    assert len(get_module(m, m.gradcam_target)._forward_hooks) == 0


def test_normalization_and_overlays(tmp_path):
    assert normalize_maps(np.ones((1, 4, 4))).max() == 0
    hm = np.random.default_rng(0).random((16, 16))
    assert normalize_heatmap(hm).max() == pytest.approx(1)
    img = np.random.default_rng(1).random((64, 64))
    ov = overlay_heatmap(img, hm)
    assert ov.shape == (64, 64, 3) and ov.dtype == np.uint8
    boxed = draw_boxes(ov, np.array([[5, 5, 20, 20], [-1, -1, -1, -1]]))
    assert boxed.shape == ov.shape
    assert isinstance(encode_png_base64(boxed), str)
    comparison_panel(img, {"Grad-CAM++": hm, "Localization": hm}, np.array([[5, 5, 20, 20]]), title="t",
                     probability=0.3, path=str(tmp_path / "p.png"))
    assert (tmp_path / "p.png").exists()
