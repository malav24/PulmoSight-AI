import pytest
import torch
import torch.nn.functional as F

from ml.training.losses import (ExplanationGuidedLoss, attention_kl, attention_outside_mass, focal_loss_with_logits,
                                mask_to_grid, soft_dice_loss)


def _batch(B=2, H=32):
    mask = torch.zeros(B, 1, H, H)
    mask[0, :, 0:16, 0:16] = 1  # image 0 positive with box in the top-left quadrant
    return {"label": torch.tensor([1.0, 0.0])[:B], "mask": mask, "has_boxes": torch.tensor([True, False])[:B]}


def test_focal_reduces_to_scaled_bce_when_gamma_zero():
    logits, y = torch.randn(10), (torch.rand(10) > 0.5).float()
    fl = focal_loss_with_logits(logits, y, alpha=0.5, gamma=0.0)
    torch.testing.assert_close(fl, 0.5 * F.binary_cross_entropy_with_logits(logits, y))


def test_dice_perfect_and_worst():
    t = torch.zeros(1, 1, 8, 8); t[..., :4, :4] = 1
    assert soft_dice_loss(t * 40 - 20, t) < 1e-3
    # predicting exactly the complement: intersection 0 -> loss = 1 - eps/(|P|+|T|+eps) = 1 - 1/65
    assert soft_dice_loss(20 * (1 - 2 * t), t).item() == pytest.approx(1 - 1 / 65, abs=1e-3)


def test_attention_outside_mass_and_kl():
    mask = torch.zeros(1, 1, 32, 32); mask[..., :16, :16] = 1
    cov = mask_to_grid(mask, (2, 2))
    inside = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    outside = torch.tensor([[[[0.0, 0.5], [0.5, 0.0]]]])
    assert attention_outside_mass(inside, cov).item() == pytest.approx(0.0)
    assert attention_outside_mass(outside, cov).item() == pytest.approx(1.0)
    assert attention_kl(inside, cov).item() == pytest.approx(0.0, abs=1e-6)
    assert attention_kl(outside, cov).item() > 5


def test_total_loss_composition_and_weights():
    b = _batch()
    out = {"logits": torch.tensor([2.0, -1.0]), "loc_logits": torch.randn(2, 1, 32, 32),
           "attention": torch.softmax(torch.randn(2, 16), 1).view(2, 1, 4, 4)}
    l0 = ExplanationGuidedLoss({"alpha": 0, "lambda": 0})(out, b)
    lb = ExplanationGuidedLoss({"alpha": 1.0, "lambda": 0})(out, b)
    lc = ExplanationGuidedLoss({"alpha": 1.0, "lambda": 0.5})(out, b)
    assert l0["loss_loc"] == 0 and l0["loss_attn"] == 0
    torch.testing.assert_close(lb["loss"], lb["loss_cls"] + lb["loss_loc"])
    torch.testing.assert_close(lc["loss"], lc["loss_cls"] + lc["loss_loc"] + 0.5 * lc["loss_attn"])
    assert lc["loss_attn"] > 0


def test_weighted_bce_uses_pos_weight():
    b = _batch()
    out = {"logits": torch.tensor([0.0, 0.0])}
    unweighted = ExplanationGuidedLoss({"classification": "bce"})(out, b)["loss"]
    weighted = ExplanationGuidedLoss({"classification": "weighted_bce"}, pos_weight=3.0)(out, b)["loss"]
    assert weighted > unweighted


def test_attention_loss_ignores_negatives_and_has_gradient():
    b = {"label": torch.tensor([0.0]), "mask": torch.zeros(1, 1, 32, 32), "has_boxes": torch.tensor([False])}
    att = torch.softmax(torch.randn(1, 16, requires_grad=True), 1).view(1, 1, 4, 4)
    loss = ExplanationGuidedLoss({"lambda": 1.0}).attention(att, b["mask"], b["has_boxes"])
    assert loss.item() == 0
    scores = torch.randn(2, 16, requires_grad=True)
    b = _batch()
    l = ExplanationGuidedLoss({"lambda": 1.0}).attention(torch.softmax(scores, 1).view(2, 1, 4, 4), b["mask"], b["has_boxes"])
    l.backward()
    assert scores.grad[0].abs().sum() > 0 and scores.grad[1].abs().sum() == 0


def test_errors_for_missing_heads_and_bad_config():
    b = _batch()
    with pytest.raises(ValueError):
        ExplanationGuidedLoss({"alpha": 1.0})({"logits": torch.zeros(2)}, b)
    with pytest.raises(ValueError):
        ExplanationGuidedLoss({"attention": "magic"})
