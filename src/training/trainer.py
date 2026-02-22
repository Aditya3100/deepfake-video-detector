import torch
import torch.nn as nn
from tqdm import tqdm
from src.config import Config

class Trainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model.to(Config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for batch in tqdm(self.train_loader):
            frames = batch["frames"].to(Config.DEVICE)
            fft = batch["fft"].to(Config.DEVICE)
            labels = batch["label"].to(Config.DEVICE)

            outputs = self.model(frames, fft)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in self.val_loader:
                frames = batch["frames"].to(Config.DEVICE)
                fft = batch["fft"].to(Config.DEVICE)
                labels = batch["label"].to(Config.DEVICE)

                outputs = self.model(frames, fft)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()

        return total_loss / len(self.val_loader)