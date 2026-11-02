"""Compact convolutional classifier for synthetic waveform patterns."""
import torch
from torch import nn

class WaveCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1,16,9,padding=4),nn.BatchNorm1d(16),nn.ReLU(),
            nn.MaxPool1d(4),nn.Conv1d(16,32,7,padding=3),
            nn.BatchNorm1d(32),nn.ReLU(),nn.MaxPool1d(4),
            nn.Conv1d(32,48,5,padding=2),nn.ReLU(),
            nn.AdaptiveAvgPool1d(12),nn.Flatten(),nn.Dropout(.2),
            nn.Linear(48*12,3))
    def forward(self,x):
        return self.net(x)
