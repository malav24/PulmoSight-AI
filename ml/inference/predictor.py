"""Inference API (no training code). Designed to be wrapped by the Stage-2 FastAPI backend.

    from ml.inference.predictor import PneumoniaPredictor
    predictor = PneumoniaPredictor("model_b")          # raises CheckpointNotFoundError if untrained
    result = predictor.predict("path/to/image.dcm")      # JSON-serialisable dict

Every result carries a disclaimer: this is a research prototype, not a medical device.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from ml.datasets.preprocessing import Preprocessor, load_image_any, load_raster
from ml.explainability.gradcam_pp import GradCAMPlusPlus
from ml.explainability.visualization import MAP_DESCRIPTIONS, encode_png_base64, overlay_heatmap
from ml.utils.checkpoints import calibration_matches, load_calibration, load_trained_model
from ml.utils.config import checkpoint_root
from ml.utils.runtime import get_device
from .quality import ALLOWED_SUFFIXES, check_image_quality
from .uncertainty import MahalanobisOOD, abstention_decision, mc_dropout_predict

DISCLAIMER = ("Research prototype. Outputs estimate the presence of pneumonia-related lung opacity as annotated in "
              "the RSNA dataset; they are not a diagnosis and must not be used for clinical decisions.")

ImageInput = Union[str, os.PathLike, np.ndarray]


def ood_stats_path(model_key: str) -> Path:
    return checkpoint_root() / "calibration" / f"{model_key}_ood_stats.npz"


def load_model(model_key: str, checkpoint_path: Optional[str] = None, device: str = "auto"):
    """Load a TRAINED model. Never returns random/ImageNet weights."""
    return load_trained_model(model_key, checkpoint_path, device=get_device(device))


class PneumoniaPredictor:
    def __init__(self, model_key: str = "model_b", checkpoint_path: Optional[str] = None, device: str = "auto",
                 mc_samples: int = 20, abstain_std: float = 0.15, abstain_margin: float = 0.0,
                 abstain_on_ood: bool = True):
        self.device = get_device(device)
        self.model_key = model_key
        self.model, self.meta = load_trained_model(model_key, checkpoint_path, device=self.device)
        self.pre = Preprocessor(self.meta.get("preprocessing") or {"image_size": self.meta.get("image_size", 384)})
        self.warnings: list = []
        cal = load_calibration(model_key)
        if cal and not calibration_matches(cal, self.meta):
            self.warnings.append("Calibration file was fitted on a different checkpoint and is ignored; "
                                 "re-run ml.evaluation.evaluate for this checkpoint.")
            cal = None
        self.calibration = cal
        self.temperature = float(cal["temperature"]) if cal else None
        if cal and cal.get("threshold") is not None:
            self.threshold = float(cal["threshold"])
            self.threshold_source = "Youden J on the full RSNA validation split (evaluation run)"
        else:
            self.threshold = float(self.meta.get("val_threshold", 0.5))
            self.threshold_source = "Youden J on RSNA validation at the best training epoch"
        ood = MahalanobisOOD.load(ood_stats_path(model_key))
        if ood is not None and ood.checkpoint_saved_at != self.meta.get("saved_at"):
            self.warnings.append("Out-of-distribution statistics belong to a different checkpoint and are ignored.")
            ood = None
        self.ood = ood
        self.mc_samples, self.abstain_std, self.abstain_margin = mc_samples, abstain_std, abstain_margin
        self.abstain_on_ood = abstain_on_ood
        loss_cfg = self.meta.get("loss_config", {})
        self.localization_trained = bool(getattr(self.model, "has_localization", False)) and float(loss_cfg.get("alpha", 0)) > 0
        self.attention_guided = float(loss_cfg.get("lambda", 0)) > 0

    # ---------------------------------------------------------------- input
    def load_input(self, image: ImageInput):
        if isinstance(image, np.ndarray):
            raw = image.astype(np.float32)
            if raw.ndim == 3:
                spread = float(np.abs(raw - raw.mean(axis=2, keepdims=True)).mean())
                raw = raw.mean(axis=2)
            else:
                spread = 0.0
            meta = {"source": "array", "rows": raw.shape[0], "cols": raw.shape[1], "color_spread": spread}
        else:
            path = Path(image)
            if path.suffix.lower() not in ALLOWED_SUFFIXES:
                raise ValueError(f"Unsupported file type {path.suffix!r}; allowed {sorted(ALLOWED_SUFFIXES)}")
            raw, meta = load_image_any(path)
        quality = check_image_quality(raw, meta)
        img_u8, geom = self.pre(raw)
        x = self.pre.to_tensor(img_u8)[None].to(self.device)
        self.last_geometry = geom  # lets callers map reference boxes onto the preprocessed grid
        return x, img_u8, quality, meta

    # ---------------------------------------------------------------- components
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.model.eval()
        return self.model(x)

    def generate_localization_map(self, x: torch.Tensor) -> Optional[np.ndarray]:
        if not self.localization_trained:
            return None
        return self.forward(x)["heatmap"][0, 0].float().cpu().numpy()

    def generate_attention_map(self, x: torch.Tensor) -> Optional[np.ndarray]:
        out = self.forward(x)
        return out["attention_map"][0, 0].float().cpu().numpy() if "attention_map" in out else None

    def generate_gradcam(self, x: torch.Tensor) -> np.ndarray:
        return GradCAMPlusPlus(self.model)(x)[0]

    def calculate_uncertainty(self, x: torch.Tensor) -> Dict[str, float]:
        r = mc_dropout_predict(self.model, x, self.mc_samples, self.temperature or 1.0)
        return {k: (float(v[0]) if isinstance(v, np.ndarray) else v) for k, v in r.items()}

    # ---------------------------------------------------------------- main
    def predict(self, image: ImageInput, explain: bool = True, gradcam: bool = True, uncertainty: bool = True,
                return_images: bool = False) -> Dict[str, Any]:
        x, img_u8, quality, meta = self.load_input(image)
        out = self.forward(x)
        logit = float(out["logits"][0])
        prob = float(out["probs"][0])
        cal_prob = float(1 / (1 + np.exp(-logit / self.temperature))) if self.temperature else None
        unc = self.calculate_uncertainty(x) if uncertainty else None
        ood_info: Dict[str, Any] = {"available": self.ood is not None}
        ood_flag = False
        if self.ood is not None and "embedding" in out:
            score = float(self.ood.score(out["embedding"].float().cpu().numpy())[0])
            ood_flag = score > self.ood.threshold
            ood_info.update({"mahalanobis_distance": score, "threshold": self.ood.threshold,
                             "threshold_percentile": self.ood.percentile, "flag": ood_flag})
        decision = abstention_decision(prob, unc["std"] if unc else None, self.threshold, self.abstain_std,
                                       self.abstain_margin, quality["passed"], ood_flag, self.abstain_on_ood)
        result: Dict[str, Any] = {
            "model": self.model_key,
            "checkpoint": self.meta.get("checkpoint_path"),
            "checkpoint_epoch": self.meta.get("epoch"),
            "logit": logit,
            "probability": prob,
            "calibrated_probability": cal_prob,
            "temperature": self.temperature,
            "threshold": self.threshold,
            "threshold_source": self.threshold_source,
            "configuration_warnings": list(self.warnings),
            "prediction": None if decision["abstain"] else int(prob >= self.threshold),
            "prediction_label": ("abstained" if decision["abstain"] else
                                 ("pneumonia-related opacity suspected" if prob >= self.threshold else
                                  "no pneumonia-related opacity detected")),
            "abstention": decision,
            "uncertainty": unc,
            "ood": ood_info,
            "quality": quality,
            "input_meta": {k: v for k, v in meta.items() if k != "color_spread"},
            "preprocessing_geometry": self.last_geometry.to_dict(),
            "maps": {},
            "map_descriptions": MAP_DESCRIPTIONS,
            "localization_trained": self.localization_trained,
            "attention_guided": self.attention_guided,
            "disclaimer": DISCLAIMER,
        }
        maps: Dict[str, np.ndarray] = {}
        if explain:
            loc = self.generate_localization_map(x)
            if loc is not None:
                maps["localization"] = loc
            if "attention_map" in out:
                maps["attention"] = out["attention_map"][0, 0].float().cpu().numpy()
        if gradcam:
            maps["gradcam"] = self.generate_gradcam(x)
        for name, m in maps.items():
            entry: Dict[str, Any] = {"shape": list(m.shape), "min": float(m.min()), "max": float(m.max()),
                                     "array": m}
            if return_images:
                entry["overlay_png_base64"] = encode_png_base64(overlay_heatmap(img_u8, m))
            result["maps"][name] = entry
        if return_images:
            result["preprocessed_png_base64"] = encode_png_base64(np.repeat(img_u8[..., None], 3, axis=2))
        return result


def to_json_safe(result: Dict[str, Any], include_arrays: bool = False) -> Dict[str, Any]:
    """Drop / convert numpy arrays so the result can be returned by an API."""
    def conv(o):
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items() if include_arrays or k != "array"}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    return conv(result)
