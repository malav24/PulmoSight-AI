import json
import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from ml.datasets.loaders import build_nih_dataset, build_rsna_datasets, make_loader
from ml.datasets.nih import load_rsna_overlap_ids, parse_nih_entries, select_nih_external_set
from ml.datasets.rsna import RSNAPneumoniaDataset, compute_rsna_statistics, make_patient_splits, parse_rsna_labels
from ml.datasets.transforms import apply_augmentation, build_train_augmentation


def test_parse_rsna_labels(rsna_root):
    patients, boxes = parse_rsna_labels(rsna_root / "stage_2_train_labels.csv",
                                        rsna_root / "stage_2_detailed_class_info.csv")
    assert patients.patientId.is_unique
    assert (patients.target == (patients.n_boxes > 0)).all()
    assert len(boxes) == patients.n_boxes.sum()
    assert (boxes.x2 > boxes.x1).all() and (boxes.y2 > boxes.y1).all()
    assert set(patients["class"]) <= {"Lung Opacity", "Normal", "No Lung Opacity / Not Normal"}


def test_patient_level_splits_are_disjoint(rsna_root):
    patients, _ = parse_rsna_labels(rsna_root / "stage_2_train_labels.csv")
    s = make_patient_splits(patients, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=1, stratify_on="target")
    groups = {k: set(g.patientId) for k, g in s.groupby("split")}
    assert not (groups["train"] & groups["val"]) and not (groups["train"] & groups["test"]) \
        and not (groups["val"] & groups["test"])
    assert sum(len(v) for v in groups.values()) == len(patients)
    s2 = make_patient_splits(patients, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=1, stratify_on="target")
    pd.testing.assert_frame_equal(s, s2)  # reproducible
    with pytest.raises(ValueError):
        make_patient_splits(patients, {"train": 0.6, "val": 0.2, "test": 0.3})


def test_rsna_statistics(rsna_root):
    patients, boxes = parse_rsna_labels(rsna_root / "stage_2_train_labels.csv")
    st = compute_rsna_statistics(patients, boxes, (128, 128))
    assert st["n_images"] == 30 and st["n_positive"] == 12
    assert 0 < st["box_relative_area"]["mean"] < 1


def test_rsna_dataset_item_mask_matches_boxes(make_cfg):
    cfg = make_cfg("model_b.yaml", data={"augmentation": {"enabled": False}})
    ds = build_rsna_datasets(cfg, splits=("train",))["train"]
    pos_idx = int(np.where(ds.labels > 0)[0][0])
    item = ds[pos_idx]
    assert item["image"].shape == (3, 64, 64) and item["mask"].shape == (1, 64, 64)
    assert item["has_boxes"] and item["n_boxes"] >= 1
    # every box region is inside the mask and the mask has nothing outside the boxes
    from ml.datasets.preprocessing import boxes_to_mask
    boxes = item["boxes"][: item["n_boxes"]].numpy()
    np.testing.assert_array_equal(boxes_to_mask(boxes, 64, 64), item["mask"][0].numpy())
    neg = ds[int(np.where(ds.labels == 0)[0][0])]
    assert neg["mask"].sum() == 0 and not neg["has_boxes"] and (neg["boxes"] == -1).all()


def test_augmentation_keeps_mask_and_boxes_aligned():
    aug = build_train_augmentation({"enabled": True, "rotate_limit": 0, "scale_limit": 0.1, "shift_limit": 0.1,
                                    "p_affine": 1.0, "brightness_contrast": 0.0})
    img = np.zeros((64, 64), np.uint8)
    boxes = np.array([[16, 20, 40, 44]], np.float32)
    from ml.datasets.preprocessing import boxes_to_mask
    mask = boxes_to_mask(boxes, 64, 64)
    for seed in range(5):
        import random
        random.seed(seed); np.random.seed(seed)
        _, m2, b2 = apply_augmentation(aug, img, mask, boxes)
        ys, xs = np.where(m2 > 0.5)
        assert len(b2) == 1
        assert abs(xs.min() - b2[0, 0]) <= 1.5 and abs(xs.max() + 1 - b2[0, 2]) <= 1.5
        assert abs(ys.min() - b2[0, 1]) <= 1.5 and abs(ys.max() + 1 - b2[0, 3]) <= 1.5


def test_rsna_dataloader_batches(make_cfg):
    cfg = make_cfg("model_b.yaml")
    ds = build_rsna_datasets(cfg)
    batch = next(iter(make_loader(ds["train"], cfg, train=True)))
    assert batch["image"].shape == (4, 3, 64, 64)
    assert batch["boxes"].shape == (4, 8, 4) and batch["has_boxes"].dtype == torch.bool
    cfg["loader"]["balanced_sampling"] = True
    assert next(iter(make_loader(ds["train"], cfg, train=True)))["image"].shape[0] == 4


def test_rsna_cache_roundtrip(make_cfg, tmp_path):
    import cv2
    cfg = make_cfg("model_b.yaml", data={"augmentation": {"enabled": False}})
    ds = build_rsna_datasets(cfg, splits=("val",))["val"]
    cache = tmp_path / "cache"
    cache.mkdir()
    recs = ds.records.copy()
    recs["orig_h"], recs["orig_w"] = 128, 128
    from ml.datasets.preprocessing import load_dicom
    for pid in recs.patientId:
        img, _ = ds.pre(load_dicom(ds.image_dir / f"{pid}.dcm")[0])
        cv2.imwrite(str(cache / f"{pid}.png"), img)
    with pytest.raises(FileNotFoundError):
        RSNAPneumoniaDataset(ds.root, recs, pd.DataFrame(columns=["patientId", "x1", "y1", "x2", "y2"]),
                             cfg["data"]["preprocessing"], cache_dir=cache)
    (cache / "cache_meta.json").write_text(json.dumps({"preprocessing": ds.pre.signature()}))
    _, boxes = parse_rsna_labels(ds.root / "stage_2_train_labels.csv")
    cached = RSNAPneumoniaDataset(ds.root, recs, boxes, cfg["data"]["preprocessing"], cache_dir=cache)
    a, b = ds[0], cached[0]
    torch.testing.assert_close(a["image"], b["image"])
    torch.testing.assert_close(a["mask"], b["mask"])


def test_nih_parsing_and_selection(nih_root):
    df = parse_nih_entries(nih_root / "Data_Entry_2017_v2020.csv")
    assert df.pneumonia.sum() == 4 and {"orig_w", "orig_h"} <= set(df.columns)
    cfg = {"subset": "official_test", "negative_definition": "no_finding", "exclude_rsna_overlap": False}
    sel = select_nih_external_set(df, cfg, nih_root)
    assert set(sel.pneumonia.astype(bool) | sel.no_finding.astype(bool)) == {True}


def test_nih_overlap_exclusion_and_warning(nih_root, tmp_path):
    df = parse_nih_entries(nih_root / "Data_Entry_2017_v2020.csv")
    mapping = tmp_path / "map.json"
    mapping.write_text(json.dumps([{"subset_img_id": "c0ffee00-0000-4000-8000-000000000001",
                                    "img_id": df.image_index.iloc[0].replace(".png", "")}]))
    assert load_rsna_overlap_ids(mapping) == {df.image_index.iloc[0]}
    sel = select_nih_external_set(df, {"subset": "all", "rsna_mapping_json": str(mapping)}, nih_root)
    assert df.image_index.iloc[0] not in set(sel.image_index) and len(sel) == len(df) - 1
    with pytest.warns(UserWarning, match="NOT excluded"):
        select_nih_external_set(df, {"subset": "all", "rsna_mapping_json": str(tmp_path / "none.json")}, nih_root)


def test_nih_dataset_items_use_only_official_boxes(make_cfg):
    cfg = make_cfg("model_b.yaml")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = build_nih_dataset(cfg)
    has = [bool(ds[i]["has_boxes"]) for i in range(len(ds))]
    assert sum(has) == 1  # only the single official synthetic bbox; nothing fabricated
    item = ds[0]
    assert item["image"].shape == (3, 64, 64) and item["dataset"] == "nih"
