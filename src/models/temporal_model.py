import torch.nn as nn
from src.config import Config

class TemporalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=Config.SPATIAL_FEATURE_DIM,
            hidden_size=Config.TEMPORAL_HIDDEN_DIM,
            batch_first=True
        )

    def forward(self, x):
        # x: (B, T, feature_dim)
        _, (hidden, _) = self.lstm(x)
        return hidden[-1]