import torch.nn as nn
from src.config import Config

class FrequencyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Linear(32, Config.FREQ_FEATURE_DIM)

    def forward(self, x):
        # x: (B*T, 1, H, W)
        x = self.cnn(x).flatten(1)
        x = self.fc(x)
        return x