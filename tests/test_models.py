import pytest
import torch

from ml.models.factory import build_model, count_parameters
from ml.utils.config import load_config

BASELINES = ["resnet50", "densenet121", "efficientnet_b0", "vit_small"]


def hybrid(size=128, **kw):
    cfg = dict(load_config("hybrid_common.yaml")["model"], pretrained_backbone=False, **kw)
    return build_model(cfg, image_size=size)


def test_hybrid_forward_shapes_and_outputs():
    torch.manual_seed(0)
    m = hybrid(128).eval()
    x = torch.randn(2, 3, 128, 128)
    out = m(x, return_raw_attention=True)
    assert out["logits"].shape == (2,)
    assert torch.all((out["probs"] >= 0) & (out["probs"] <= 1))
    assert out["loc_logits"].shape == (2, 1, 128, 128)
    assert out["heatmap"].shape == (2, 1, 128, 128)
    assert out["attention"].shape == (2, 1, 4, 4)  # 128 / 32
    assert torch.allclose(out["attention"].sum(dim=(1, 2, 3)), torch.ones(2), atol=1e-5)
    assert out["attention_map"].shape == (2, 1, 128, 128)
    assert out["attention_map"].min() >= 0 and out["attention_map"].max() <= 1 + 1e-6
    assert out["raw_self_attention"].shape == (2, 2, 16, 16)  # (B, layers, N, N)
    assert out["embedding"].shape == (2, 256)


def test_hybrid_384_token_count_and_positional_interpolation():
    m = hybrid(384).eval()
    with torch.no_grad():
        out = m(torch.randn(1, 3, 384, 384))
        assert out["attention"].shape[-2:] == (12, 12)  # 144 tokens
        out2 = m(torch.randn(1, 3, 256, 256))           # different size -> pos-embed interpolation
        assert out2["attention"].shape[-2:] == (8, 8)


def test_hybrid_token_pool_reduces_tokens():
    m = hybrid(256, token_pool=2).eval()
    with torch.no_grad():
        assert m(torch.randn(1, 3, 256, 256))["attention"].shape[-2:] == (4, 4)


def test_hybrid_gradients_reach_backbone_and_heads():
    m = hybrid(64)
    out = m(torch.randn(2, 3, 64, 64))
    (out["logits"].sum() + out["loc_logits"].mean() + out["attention"].pow(2).sum()).backward()
    assert m.backbone.conv_stem.weight.grad is not None
    assert m.attn_score[0].weight.grad is not None
    assert m.loc_decoder[-1].weight.grad is not None


def test_model_a_and_b_have_identical_architecture():
    a = build_model(dict(load_config("model_a.yaml")["model"], pretrained_backbone=False), 128)
    b = build_model(dict(load_config("model_b.yaml")["model"], pretrained_backbone=False), 128)
    c = build_model(dict(load_config("ablation_cls_only.yaml")["model"], pretrained_backbone=False), 128)
    ka, kb, kc = a.state_dict().keys(), b.state_dict().keys(), c.state_dict().keys()
    assert list(ka) == list(kb) == list(kc)
    assert count_parameters(a) == count_parameters(b)


@pytest.mark.parametrize("name", BASELINES)
def test_baseline_forward(name):
    m = build_model({"name": name, "pretrained_backbone": False}, image_size=64).eval()
    with torch.no_grad():
        out = m(torch.randn(2, 3, 64, 64))
    assert out["logits"].shape == (2,)
    assert out["probs"].shape == (2,)
    assert out["embedding"].ndim == 2 and out["embedding"].shape[0] == 2
    assert "heatmap" not in out
    assert len(m.backbone_parameters()) < len(list(m.parameters()))


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        build_model({"name": "nope"})
