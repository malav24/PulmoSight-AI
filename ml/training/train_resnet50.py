"""ResNet50 baseline (classification only).

Usage (from the project root):
    python -m ml.training.train_resnet50 [--resume] [--epochs N] [--batch-size B] [key=value ...]
"""
from ml.training.trainer import main

if __name__ == "__main__":
    main("resnet50.yaml")
