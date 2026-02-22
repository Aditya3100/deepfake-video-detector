import torch
import torch.nn as nn
from src.models.spatial_model import SpatialModel
from src.models.temporal_model import TemporalModel
from src.models.frequency_model import FrequencyModel
from src.config import Config

class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = SpatialModel()
        self.temporal = TemporalModel()
        self.frequency = FrequencyModel()

        fusion_dim = (
            Config.TEMPORAL_HIDDEN_DIM +
            Config.FREQ_FEATURE_DIM
        )

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, frames, fft):
        B, T, C, H, W = frames.shape

        frames = frames.view(B*T, C, H, W)
        fft = fft.view(B*T, 1, H, W)

        spatial_features = self.spatial(frames)
        spatial_features = spatial_features.view(B, T, -1)

        temporal_features = self.temporal(spatial_features)

        freq_features = self.frequency(fft)
        freq_features = freq_features.view(B, T, -1).mean(1)

        fused = torch.cat([temporal_features, freq_features], dim=1)
        output = self.classifier(fused)

        return output.squeeze()