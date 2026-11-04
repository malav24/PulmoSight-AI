"""Dataset / DataLoader construction from a config dict."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from ml.utils.config import resolve_path
from ml.utils.runtime import seed_worker
from .nih import (NIHChestXrayDataset, index_nih_images, load_nih_pneumonia_bboxes, parse_nih_entries,
                  select_nih_external_set)
from .rsna import RSNAPneumoniaDataset, load_or_create_splits


def build_rsna_datasets(cfg: Dict[str, Any], splits=("train", "val", "test")) -> Dict[str, RSNAPneumoniaDataset]:
    data = cfg["data"]
    rcfg = data["rsna"]
    split_df, boxes = load_or_create_splits(rcfg, data["split"], cfg["experiment"]["seed"])
    root = resolve_path(rcfg["root"])
    cache = resolve_path(rcfg.get("cache_dir")) if rcfg.get("cache_dir") else None
    out = {}
    for s in splits:
        out[s] = RSNAPneumoniaDataset(
            root, split_df[split_df.split == s], boxes, data["preprocessing"],
            augmentation=data.get("augmentation") if s == "train" else None,
            image_dir=rcfg.get("image_dir", "stage_2_train_images"), cache_dir=cache,
            max_boxes=data.get("max_boxes", 8))
    return out


def build_nih_dataset(cfg: Dict[str, Any]) -> NIHChestXrayDataset:
    data = cfg["data"]
    ncfg = data["nih"]
    root = resolve_path(ncfg["root"])
    from .nih import find_entries_csv
    entries = find_entries_csv(root, ncfg.get("entries_csv"))
    if entries is None:
        raise FileNotFoundError(f"No NIH metadata CSV (Data_Entry_2017*.csv or sample_labels.csv) in {root}")
    df = select_nih_external_set(parse_nih_entries(entries), ncfg, root, cfg["experiment"]["seed"])
    bboxes = None
    bbox_csv = root / ncfg.get("bbox_csv", "BBox_List_2017.csv")
    if ncfg.get("use_official_bboxes", True) and bbox_csv.exists():
        bboxes = load_nih_pneumonia_bboxes(bbox_csv)
    ds = NIHChestXrayDataset(root, df, data["preprocessing"], bboxes=bboxes, image_index=index_nih_images(root),
                             max_boxes=data.get("max_boxes", 8))
    ds.selection_notes = df.attrs.get("selection_notes", [])
    return ds


def make_loader(dataset, cfg: Dict[str, Any], train: bool) -> DataLoader:
    lcfg = cfg["loader"]
    bs = lcfg["batch_size"] if train else lcfg.get("eval_batch_size", lcfg["batch_size"])
    sampler = None
    shuffle = train
    if train and lcfg.get("balanced_sampling", False):
        labels = dataset.labels
        pos = max(labels.sum(), 1.0)
        neg = max(len(labels) - labels.sum(), 1.0)
        weights = np.where(labels > 0, 0.5 / pos, 0.5 / neg)
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(labels), replacement=True)
        shuffle = False
    g = torch.Generator()
    g.manual_seed(int(cfg["experiment"]["seed"]))
    nw = int(lcfg.get("num_workers", 0))
    return DataLoader(dataset, batch_size=bs, shuffle=shuffle, sampler=sampler, num_workers=nw,
                      pin_memory=bool(lcfg.get("pin_memory", False)) and torch.cuda.is_available(),
                      drop_last=train and len(dataset) > bs, worker_init_fn=seed_worker, generator=g,
                      persistent_workers=nw > 0)
