import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROGRESS_FOLDER = "progress"
TRAINING_FOLDER = "training"
