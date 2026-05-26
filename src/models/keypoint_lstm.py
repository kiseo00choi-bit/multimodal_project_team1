from __future__ import annotations

import torch
from torch import nn


class KeypointLSTMModel(nn.Module):
    def __init__(self, num_classes: int = 8, keypoint_dim: int = 34, hidden_dim: int = 128):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(keypoint_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.rnn = nn.GRU(128, hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, frames: torch.Tensor | None = None, keypoints: torch.Tensor | None = None) -> torch.Tensor:
        if keypoints is None:
            raise ValueError("keypoints are required")
        x = keypoints.transpose(1, 2)
        x = self.temporal(x).transpose(1, 2)
        output, hidden = self.rnn(x)
        return self.classifier(hidden[-1])
