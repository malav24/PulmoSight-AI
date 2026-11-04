"""Report environment, dataset presence and checkpoint status. Safe to run any time.

    python scripts/verify_setup.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.datasets.nih import find_entries_csv  # noqa: E402
from ml.utils.checkpoints import discover_checkpoints  # noqa: E402
from ml.utils.config import PROJECT_ROOT, load_config, resolve_path  # noqa: E402
from ml.utils.runtime import environment_report  # noqa: E402


def main():
    cfg = load_config("base.yaml")
    env = environment_report(str(PROJECT_ROOT))
    r, n = cfg["data"]["rsna"], cfg["data"]["nih"]
    rroot, nroot = resolve_path(r["root"]), resolve_path(n["root"])
    data = {
        "rsna_labels_csv": (rroot / r["labels_csv"]).exists(),
        "rsna_class_info_csv": (rroot / r["class_info_csv"]).exists(),
        "rsna_n_dicoms": len(list((rroot / r["image_dir"]).glob("*.dcm"))) if (rroot / r["image_dir"]).exists() else 0,
        "rsna_splits_csv": resolve_path(r["splits_csv"]).exists(),
        "nih_entries_csv": find_entries_csv(nroot, n["entries_csv"]) is not None,
        "nih_test_list": (nroot / n["test_list"]).exists(),
        "nih_bbox_csv": (nroot / n["bbox_csv"]).exists(),
        "nih_rsna_mapping_json": resolve_path(n["rsna_mapping_json"]).exists()
                                 or (nroot / "pneumonia-challenge-dataset-mappings_2018.json").exists(),
        "nih_n_pngs": sum(1 for _ in nroot.rglob("*.png")) if nroot.exists() else 0,
    }
    ck = discover_checkpoints()
    print("== Environment ==")
    print(json.dumps(env, indent=2))
    print("== Datasets ==")
    print(json.dumps(data, indent=2))
    print("== Checkpoints (best) ==")
    for k, v in ck.items():
        if k.startswith("_"):
            print(f"  calibration: {'present' if v['exists'] else 'MISSING'}  {v['path']}")
        else:
            print(f"  {k:18s} {'present' if v['best']['exists'] else 'MISSING (not trained yet)'}  {v['best']['path']}")
    if not data["rsna_labels_csv"]:
        print("\nRSNA not found in the configured folder. Run  python scripts/prepare_rsna.py --cache  - it searches your "
              "Downloads/Desktop/Documents, unzips a Kaggle zip, or downloads with --download.")
    if not env["cuda_available"]:
        print("\nNOTE: no CUDA GPU visible to PyTorch. Training at 384px on CPU is impractically slow.")


if __name__ == "__main__":
    main()
