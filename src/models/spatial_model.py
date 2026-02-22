import torch.nn as nn
import torchvision.models as models
from src.config import Config

class SpatialModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.efficientnet_b0(pretrained=True)
        self.feature_extractor = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, Config.SPATIAL_FEATURE_DIM)

    def forward(self, x):
        # x: (B*T, 3, H, W)
        x = self.feature_extractor(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x