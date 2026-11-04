"""Ablation A - hybrid, classification only.

Usage (from the project root):
    python -m ml.training.train_ablation_cls_only [--resume] [--epochs N] [--batch-size B] [key=value ...]
"""
from ml.training.trainer import main

if __name__ == "__main__":
    main("ablation_cls_only.yaml")
