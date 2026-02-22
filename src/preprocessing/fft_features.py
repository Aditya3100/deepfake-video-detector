import cv2
import numpy as np
import torch

def compute_fft(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log(np.abs(fft_shift) + 1)

    magnitude = cv2.resize(magnitude, (224, 224))
    magnitude = magnitude / np.max(magnitude)

    tensor = torch.tensor(magnitude, dtype=torch.float32).unsqueeze(0)
    return tensor