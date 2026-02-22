import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from src.preprocessing.fft_features import compute_fft
from src.config import Config

class VideoDataset(Dataset):
    def __init__(self, video_paths, labels):
        self.video_paths = video_paths
        self.labels = labels

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.video_paths)

    def sample_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames-1, Config.NUM_FRAMES).astype(int)

        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()
        return frames

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        frames = self.sample_frames(video_path)

        spatial_frames = []
        fft_frames = []

        for frame in frames:
            spatial_frames.append(self.transform(frame))
            fft_frames.append(compute_fft(frame))

        spatial_tensor = torch.stack(spatial_frames)
        fft_tensor = torch.stack(fft_frames)

        return {
            "frames": spatial_tensor,
            "fft": fft_tensor,
            "label": label
        }