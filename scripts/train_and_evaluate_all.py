"""Master pipeline to train and evaluate all remaining models on GPU,
run comparisons, generate publication figures, and update report files.
"""
from __future__ import annotations

import os
import sys
import time
import json
import logging
import subprocess
from pathlib import Path

# Setup paths and force cache to D: drive
PROJECT_ROOT = Path("D:/PulmoSight-AI-main")
os.environ["TORCH_HOME"] = str(PROJECT_ROOT / ".cache/torch")
os.environ["HF_HOME"] = str(PROJECT_ROOT / ".cache/huggingface")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
sys.path.insert(0, str(PROJECT_ROOT))

from ml.utils.config import load_config, output_root, checkpoint_root
from ml.training.trainer import run_training
from ml.evaluation.evaluate import evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pipeline")

MODELS = [
    {
        "name": "model_a",
        "cfg": "model_a.yaml",
        "key": "model_a",
        "ckpt": "hybrid/model_a_best.pth",
        "overrides": {"data.rsna.cache_dir": "data/rsna/cache_384_clahe"}
    },
    {
        "name": "ablation_cls_only",
        "cfg": "ablation_cls_only.yaml",
        "key": "ablation_cls_only",
        "ckpt": "hybrid/ablation_cls_only_best.pth",
        "overrides": {"data.rsna.cache_dir": "data/rsna/cache_384_clahe"}
    },
    {
        "name": "efficientnet_b0",
        "cfg": "efficientnet_b0.yaml",
        "key": "efficientnet_b0",
        "ckpt": "baselines/efficientnet_b0_best.pth",
        "overrides": {"data.rsna.cache_dir": "data/rsna/cache_384_clahe"}
    },
    {
        "name": "densenet121",
        "cfg": "densenet121.yaml",
        "key": "densenet121",
        "ckpt": "baselines/densenet121_best.pth",
        "overrides": {"data.rsna.cache_dir": "data/rsna/cache_384_clahe"}
    },
    {
        "name": "resnet50",
        "cfg": "resnet50.yaml",
        "key": "resnet50",
        "ckpt": "baselines/resnet50_best.pth",
        "overrides": {"data.rsna.cache_dir": "data/rsna/cache_384_clahe"}
    },
    {
        "name": "vit_small",
        "cfg": "vit_small.yaml",
        "key": "vit_small",
        "ckpt": "baselines/vit_small_best.pth",
        "overrides": {"data.rsna.cache_dir": "data/rsna/cache_384_clahe", "loader.batch_size": 8}
    }
]

def main():
    logger.info("Starting master training and evaluation pipeline...")
    # Check Model B evaluation exists
    if not (output_root() / "metrics/model_b_metrics.json").exists():
        logger.info("Evaluating model_b first...")
        evaluate("model_b", use_nih=True, explain_n=32, stress=True)

    summary = {}
    for item in MODELS:
        ckpt_path = checkpoint_root() / item["ckpt"]
        logger.info("=" * 60)
        logger.info(f"Target: {item['name']} | Checkpoint: {ckpt_path}")
        
        # 1. Train if checkpoint doesn't exist
        if not ckpt_path.exists():
            logger.info(f"Training {item['name']}...")
            t0 = time.time()
            cfg = load_config(item["cfg"], item["overrides"])
            train_res = run_training(cfg)
            elapsed = time.time() - t0
            summary[f"{item['name']}_train_seconds"] = elapsed
            logger.info(f"Finished training {item['name']} in {elapsed/60:.1f} mins.")
        else:
            logger.info(f"Checkpoint already exists for {item['name']}. Skipping training.")

        # 2. Evaluate
        metrics_path = output_root() / f"metrics/{item['key']}_metrics.json"
        logger.info(f"Evaluating {item['key']}...")
        try:
            ev_res = evaluate(item["key"], use_nih=True, explain_n=32, stress=True)
            logger.info(f"Saved evaluation metrics to {metrics_path}")
            summary[f"{item['key']}_eval"] = "success"
        except Exception as e:
            logger.error(f"Error evaluating {item['key']}: {e}")
            summary[f"{item['key']}_eval"] = str(e)

    # 3. Model Comparison (DeLong Paired Tests)
    logger.info("Running paired comparison across all models...")
    cmd_compare = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/compare_models.py"),
        "--models", "model_b", "model_a", "resnet50", "densenet121", "efficientnet_b0", "vit_small"
    ]
    subprocess.run(cmd_compare, check=False)

    # 4. Ablation Summary
    cmd_ablation = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/run_ablation.py"),
        "--summarize"
    ]
    subprocess.run(cmd_ablation, check=False)

    # Save pipeline summary
    sum_path = output_root() / "reports/pipeline_execution_summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"PIPELINE COMPLETED! Summary saved to {sum_path}")

if __name__ == "__main__":
    main()
