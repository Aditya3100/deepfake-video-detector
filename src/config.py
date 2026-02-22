import torch

class Config:
    # Data
    IMG_SIZE = 224
    NUM_FRAMES = 16
    BATCH_SIZE = 4
    NUM_WORKERS = 2

    # Model
    SPATIAL_FEATURE_DIM = 512
    TEMPORAL_HIDDEN_DIM = 256
    FREQ_FEATURE_DIM = 128

    # Training
    LR = 1e-4
    EPOCHS = 10
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    TRAIN_DIR = "data/train"
    VAL_DIR = "data/val"
    CHECKPOINT_DIR = "checkpoints/"