"""Evaluate one TRAINED model end-to-end.

    python -m ml.evaluation.evaluate --model model_b [--nih] [--explain-n 64] [--stress] [--fit-ood]

Steps
  1. Load the trained checkpoint (fails clearly if absent).
  2. Predict on RSNA val + test (and NIH if requested); save per-image predictions CSVs.
  3. Fit temperature scaling on RSNA VAL logits only; apply to test / NIH; save temperature.
  4. Threshold = Youden J on RSNA VAL; applied unchanged to test / NIH.
  5. Classification metrics + bootstrap and DeLong CIs; calibration (ECE, Brier, reliability diagram).
  6. Hybrid models: localization (IoU/Dice/pointing game) and attention alignment on annotated images.
  7. Grad-CAM++ on a subset of RSNA test positives: localization + deletion/insertion AUC.
  8. Shortcut stress tests (optional) and RSNA-vs-NIH generalisation gap.
Nothing here trains or updates model weights.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from ml.utils.checkpoints import checkpoint_identity, load_trained_model, save_calibration
from ml.utils.config import apply_env_overrides, deep_merge, output_root
from ml.utils.logging_utils import get_logger, save_json
from ml.utils.runtime import get_device, set_seed
from .calibration import TemperatureScaler, calibration_report, plot_reliability_diagram
from .classification import classification_metrics, plot_roc, safe_auc
from .explanation import attention_alignment, explanation_metrics
from .generalization import generalization_report
from .localization import mass_in_box, pointing_game_hit, dice, iou
from .statistics import bootstrap_ci, delong_auc_ci

log = get_logger("rfrr.eval")


@torch.no_grad()
def collect_predictions(model, loader, device, max_batches: Optional[int] = None, localization: bool = False
                        ) -> Dict[str, Any]:
    model.eval()
    rows: List[Dict[str, Any]] = []
    loc_rows: List[Dict[str, Any]] = []
    attn_parts: List[Dict[str, float]] = []
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x = batch["image"].to(device)
        out = model(x)
        logits = out["logits"].float().cpu().numpy()
        probs = out["probs"].float().cpu().numpy()
        for j in range(len(logits)):
            rows.append({"id": batch["id"][j], "label": float(batch["label"][j]), "logit": float(logits[j]),
                         "prob": float(probs[j]),
                         "patient_id": batch["patient_id"][j] if "patient_id" in batch else batch["id"][j]})
        has = batch["has_boxes"].bool()
        if has.any() and "attention" in out:
            r = attention_alignment(out["attention"][has.to(device)].cpu(), batch["mask"][has])
            if r.get("n"):
                attn_parts.append(r)
        if localization and has.any() and "heatmap" in out:
            hms = out["heatmap"][:, 0].float().cpu().numpy()
            masks = batch["mask"][:, 0].numpy()
            for j in np.where(has.numpy())[0]:
                pm = (hms[j] >= 0.5).astype(np.float32)
                loc_rows.append({"id": batch["id"][j], "iou": iou(pm, masks[j]), "dice": dice(pm, masks[j]),
                                 "pointing_game": float(pointing_game_hit(hms[j], masks[j])),
                                 "mass_in_box": mass_in_box(hms[j], masks[j])})
    attn = {}
    if attn_parts:
        n = sum(p["n"] for p in attn_parts)
        attn = {k: float(sum(p[k] * p["n"] for p in attn_parts) / n) for k in attn_parts[0] if k != "n"}
        attn["n"] = n
    return {"predictions": pd.DataFrame(rows), "localization": pd.DataFrame(loc_rows), "attention": attn}


def summarize_classification(df: pd.DataFrame, threshold: float, prob_col: str = "prob", groups=None,
                             n_boot: int = 2000) -> Dict[str, Any]:
    y, p = df["label"].to_numpy(), df[prob_col].to_numpy()
    res = classification_metrics(y, p, threshold)
    if len(np.unique(y)) == 2:
        res["roc_auc_bootstrap_ci"] = bootstrap_ci(safe_auc, y, p, n_boot=n_boot, groups=groups)
        res["roc_auc_delong_ci"] = delong_auc_ci(y, p)
    return res


def gradcam_analysis(model, dataset, device, n: int, steps: int, out_dir: Path, model_key: str) -> Dict[str, Any]:
    from ml.explainability.gradcam_pp import GradCAMPlusPlus
    from ml.explainability.visualization import comparison_panel
    from .localization import localization_metrics

    idx = [i for i in range(len(dataset)) if float(dataset.records.iloc[i]["target"]) == 1][:n]
    if not idx:
        return {"n": 0}
    cam = GradCAMPlusPlus(model)
    sal, masks, imgs = [], [], []
    for start in range(0, len(idx), 8):
        items = [dataset[i] for i in idx[start:start + 8]]
        x = torch.stack([it["image"] for it in items]).to(device)
        sal.append(cam(x))
        masks.append(np.stack([it["mask"][0].numpy() for it in items]))
        imgs.append(x.cpu())
    sal, masks, imgs = np.concatenate(sal), np.concatenate(masks), torch.cat(imgs)

    def predict_fn(t):
        with torch.no_grad():
            return model(t.to(device))["probs"]

    res = {"gradcam_localization": localization_metrics(list(sal), list(masks), threshold=0.5)}
    res["gradcam_faithfulness"] = explanation_metrics(predict_fn, imgs, sal, masks, steps=steps)
    # A few qualitative panels
    pre = dataset.pre
    for k in range(min(4, len(idx))):
        it = dataset[idx[k]]
        with torch.no_grad():
            out = model(it["image"][None].to(device))
        maps = {"Grad-CAM++": sal[k]}
        if "heatmap" in out:
            maps["Localization"] = out["heatmap"][0, 0].float().cpu().numpy()
        if "attention_map" in out:
            maps["Attention"] = out["attention_map"][0, 0].float().cpu().numpy()
        boxes = it["boxes"].numpy()[: int(it["n_boxes"])]
        comparison_panel(pre.denormalize(it["image"]), maps, boxes, title=f"{model_key} | {it['id'][:8]}",
                         probability=float(out["probs"][0]), path=str(out_dir / f"{model_key}_panel_{k}.png"))
    return res


def evaluate(model_key: str, checkpoint: Optional[str] = None, use_nih: bool = False, explain_n: int = 32,
             explain_steps: int = 20, stress: bool = False, fit_ood: bool = False, max_batches: Optional[int] = None,
             n_boot: int = 2000, config_overrides: Optional[Dict[str, Any]] = None, device: str = "auto") -> Dict[str, Any]:
    from ml.datasets.loaders import build_nih_dataset, build_rsna_datasets, make_loader

    dev = get_device(device)
    model, meta = load_trained_model(model_key, checkpoint, device=dev)
    cfg = apply_env_overrides(meta["config"])  # dataset roots may have moved since training
    if config_overrides:
        cfg = deep_merge(cfg, config_overrides)
    set_seed(int(cfg["experiment"]["seed"]))
    loss_cfg = meta.get("loss_config", {})
    loc_trained = getattr(model, "has_localization", False) and float(loss_cfg.get("alpha", 0)) > 0

    pred_dir, met_dir, vis_dir = (output_root() / d for d in ("predictions", "metrics", "visualizations"))
    for d in (pred_dir, met_dir, vis_dir):
        d.mkdir(parents=True, exist_ok=True)

    ds = build_rsna_datasets(cfg, splits=("val", "test"))
    results: Dict[str, Any] = {"model": model_key, "checkpoint": meta["checkpoint_path"], "epoch": meta.get("epoch"),
                               "checkpoint_saved_at": meta.get("saved_at"),
                               "localization_head_trained": bool(loc_trained),
                               "attention_guided": float(loss_cfg.get("lambda", 0)) > 0}
    collected = {}
    for split in ("val", "test"):
        collected[split] = collect_predictions(model, make_loader(ds[split], cfg, train=False), dev, max_batches,
                                               localization=loc_trained)
    # Calibration + threshold from VALIDATION only
    val_df = collected["val"]["predictions"]
    scaler = TemperatureScaler().fit(val_df.logit.to_numpy(), val_df.label.to_numpy())
    from ml.training.trainer import youden_threshold
    threshold = youden_threshold(val_df.label.to_numpy(), val_df.prob.to_numpy())
    if scaler.at_bound:
        log.warning("Temperature hit its bound (%.3f): calibration unreliable (check validation data/model)",
                    scaler.temperature)
    save_calibration(model_key, {"temperature": scaler.temperature, "temperature_at_bound": scaler.at_bound,
                                 "threshold": threshold,
                                 "threshold_method": "Youden J on RSNA validation", "fitted_on": "rsna_val",
                                 "n": int(len(val_df)), **checkpoint_identity(meta)})
    results["threshold"] = {"value": threshold, "source": "Youden J on RSNA validation"}
    results["temperature"] = scaler.temperature
    results["temperature_at_bound"] = scaler.at_bound

    curves_roc, curves_rel = {}, {}
    for split in ("val", "test"):
        df = collected[split]["predictions"]
        df["prob_calibrated"] = scaler.predict_proba(df.logit.to_numpy())
        df.to_csv(pred_dir / f"{model_key}_rsna_{split}.csv", index=False)
    test_df = collected["test"]["predictions"]
    results["rsna_test"] = {
        "classification": summarize_classification(test_df, threshold, n_boot=n_boot),
        "calibration_uncalibrated": calibration_report(test_df.label, test_df.prob),
        "calibration_temperature_scaled": calibration_report(test_df.label, test_df.prob_calibrated),
    }
    if loc_trained and len(collected["test"]["localization"]):
        loc = collected["test"]["localization"]
        results["rsna_test"]["localization_heatmap"] = {"n": int(len(loc)), "threshold": 0.5,
                                                        **{c: float(loc[c].mean()) for c in ("iou", "dice", "pointing_game", "mass_in_box")}}
        loc.to_csv(pred_dir / f"{model_key}_rsna_test_localization.csv", index=False)
    if collected["test"]["attention"]:
        results["rsna_test"]["attention_alignment"] = collected["test"]["attention"]
    curves_roc["RSNA test"] = (test_df.label.to_numpy(), test_df.prob.to_numpy())
    curves_rel["RSNA test (raw)"] = (test_df.label.to_numpy(), test_df.prob.to_numpy())
    curves_rel["RSNA test (temp-scaled)"] = (test_df.label.to_numpy(), test_df.prob_calibrated.to_numpy())

    if explain_n > 0:
        log.info("Grad-CAM++ analysis on %d RSNA test positives", explain_n)
        results["rsna_test"]["explanations"] = gradcam_analysis(model, ds["test"], dev, explain_n, explain_steps,
                                                               vis_dir, model_key)

    if stress:
        from .shortcut import run_stress_tests

        def predict_fn(t):
            return model(t)["probs"]
        results["rsna_test"]["stress_tests"] = run_stress_tests(
            predict_fn, make_loader(ds["test"], cfg, train=False), dev, threshold, max_batches)

    if use_nih:
        nih = build_nih_dataset(cfg)
        res_nih = collect_predictions(model, make_loader(nih, cfg, train=False), dev, max_batches, localization=loc_trained)
        ndf = res_nih["predictions"]
        ndf["prob_calibrated"] = scaler.predict_proba(ndf.logit.to_numpy())
        ndf.to_csv(pred_dir / f"{model_key}_nih.csv", index=False)
        groups = ndf.patient_id.to_numpy()
        results["nih_external"] = {
            "selection_notes": getattr(nih, "selection_notes", []),
            "classification": summarize_classification(ndf, threshold, groups=groups, n_boot=n_boot),
            "calibration_temperature_scaled": calibration_report(ndf.label, ndf.prob_calibrated),
        }
        if len(res_nih["localization"]):
            loc = res_nih["localization"]
            results["nih_external"]["localization_official_boxes"] = {
                "n": int(len(loc)), **{c: float(loc[c].mean()) for c in ("iou", "dice", "pointing_game", "mass_in_box")}}
        results["generalization"] = generalization_report(test_df.label.to_numpy(), test_df.prob.to_numpy(),
                                                          ndf.label.to_numpy(), ndf.prob.to_numpy(), groups, n_boot)
        curves_roc["NIH external"] = (ndf.label.to_numpy(), ndf.prob.to_numpy())
        curves_rel["NIH (temp-scaled)"] = (ndf.label.to_numpy(), ndf.prob_calibrated.to_numpy())

    if fit_ood:
        results["ood"] = fit_ood_statistics(model, model_key, {**cfg, "_ckpt_saved_at": meta.get("saved_at")}, dev,
                                            max_batches)

    plot_roc(curves_roc, str(vis_dir / f"{model_key}_roc.png"), f"ROC - {model_key}")
    plot_reliability_diagram(curves_rel, str(vis_dir / f"{model_key}_reliability.png"), title=f"Reliability - {model_key}")
    out_path = save_json(results, met_dir / f"{model_key}_metrics.json")
    log.info("Saved metrics to %s", out_path)
    return results


@torch.no_grad()
def fit_ood_statistics(model, model_key: str, cfg, device, max_batches=None) -> Dict[str, Any]:
    from ml.datasets.loaders import build_rsna_datasets, make_loader
    from ml.inference.predictor import ood_stats_path
    from ml.inference.uncertainty import MahalanobisOOD

    ds = build_rsna_datasets(deep_merge(cfg, {"data": {"augmentation": {"enabled": False}}}), splits=("train", "val"))

    def embed(loader):
        out = []
        for i, b in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            out.append(model(b["image"].to(device))["embedding"].float().cpu().numpy())
        return np.concatenate(out)

    tr = embed(make_loader(ds["train"], cfg, train=False))
    va = embed(make_loader(ds["val"], cfg, train=False))
    ood = MahalanobisOOD.fit(tr, va, percentile=99.0)
    ood.checkpoint_saved_at = cfg.get("_ckpt_saved_at")
    ood.save(ood_stats_path(model_key))
    return {"path": str(ood_stats_path(model_key)), "threshold": ood.threshold, "n_train": len(tr), "n_val": len(va)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate a trained model")
    ap.add_argument("--model", required=True, help="model key, e.g. model_b, resnet50")
    ap.add_argument("--checkpoint", help="explicit checkpoint path (default: registry best)")
    ap.add_argument("--nih", action="store_true", help="also evaluate on NIH ChestX-ray14 (external)")
    ap.add_argument("--explain-n", type=int, default=32)
    ap.add_argument("--explain-steps", type=int, default=20)
    ap.add_argument("--stress", action="store_true", help="run shortcut stress tests")
    ap.add_argument("--fit-ood", action="store_true", help="fit Mahalanobis OOD statistics on RSNA train/val")
    ap.add_argument("--max-batches", type=int)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    evaluate(a.model, a.checkpoint, a.nih, a.explain_n, a.explain_steps, a.stress, a.fit_ood, a.max_batches,
             a.n_boot, device=a.device)


if __name__ == "__main__":
    main()
