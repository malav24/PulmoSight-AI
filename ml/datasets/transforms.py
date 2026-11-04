"""Training augmentation that keeps image, mask and boxes geometrically consistent.

The mask is rasterised from boxes *before* augmentation and transformed with the image, so its
pixels stay aligned exactly. Boxes are transformed by albumentations; after a rotation the
returned boxes are the axis-aligned hull of the rotated box (slightly larger) - the mask is the
authoritative localisation target, boxes are kept for the pointing game / reporting.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


def build_train_augmentation(cfg: Dict[str, Any]):
    """Return an albumentations Compose, or None if augmentation is disabled."""
    if not cfg or not cfg.get("enabled", True):
        return None
    import albumentations as A

    transforms = []
    if cfg.get("horizontal_flip", False):
        transforms.append(A.HorizontalFlip(p=0.5))
    rot = float(cfg.get("rotate_limit", 7))
    scale = float(cfg.get("scale_limit", 0.08))
    shift = float(cfg.get("shift_limit", 0.05))
    affine_kwargs = dict(scale=(1 - scale, 1 + scale), translate_percent=(-shift, shift), rotate=(-rot, rot),
                         interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST,
                         border_mode=cv2.BORDER_CONSTANT, p=float(cfg.get("p_affine", 0.7)))
    try:
        transforms.append(A.Affine(fill=0, fill_mask=0, **affine_kwargs))
    except TypeError:  # albumentations < 2.0 naming
        affine_kwargs.pop("border_mode")
        transforms.append(A.Affine(mode=cv2.BORDER_CONSTANT, cval=0, cval_mask=0, **affine_kwargs))
    bc = float(cfg.get("brightness_contrast", 0.15))
    if bc > 0:
        transforms.append(A.RandomBrightnessContrast(brightness_limit=bc, contrast_limit=bc,
                                                     p=float(cfg.get("p_intensity", 0.5))))
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["box_labels"], min_visibility=0.2, clip=True),
    )


def apply_augmentation(aug, image: np.ndarray, mask: np.ndarray, boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if aug is None:
        return image, mask, boxes
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    out = aug(image=image, mask=mask, bboxes=[tuple(b) for b in boxes], box_labels=[1] * len(boxes))
    new_boxes = np.asarray(out["bboxes"], dtype=np.float32).reshape(-1, 4)
    return out["image"], out["mask"].astype(np.float32), new_boxes


def pad_boxes(boxes: np.ndarray, max_boxes: int) -> Tuple[np.ndarray, int]:
    """Pad (N,4) boxes to (max_boxes,4) with -1; returns (padded, n_valid)."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)[:max_boxes]
    out = -np.ones((max_boxes, 4), dtype=np.float32)
    out[: len(boxes)] = boxes
    return out, len(boxes)
