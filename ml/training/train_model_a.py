"""Hybrid Model A - classification + localization, no attention guidance.

Usage (from the project root):
    python -m ml.training.train_model_a [--resume] [--epochs N] [--batch-size B] [key=value ...]
"""
from ml.training.trainer import main

if __name__ == "__main__":
    main("model_a.yaml")
