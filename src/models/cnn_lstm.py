from __future__ import annotations

import torch
from torch import nn

from src.models.cnn_avg import resnet18_encoder


class CNNLSTMModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        hidden_dim: int = 256,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.encoder, feature_dim = resnet18_encoder(pretrained=pretrained)
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.rnn = nn.GRU(feature_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.encoder.eval()
        return self

    def forward(self, frames: torch.Tensor, keypoints: torch.Tensor | None = None) -> torch.Tensor:
        bsz, steps, channels, height, width = frames.shape
        x = frames.reshape(bsz * steps, channels, height, width)
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.encoder(x).reshape(bsz, steps, -1)
        else:
            features = self.encoder(x).reshape(bsz, steps, -1)
        features = self.feature_norm(features)
        output, hidden = self.rnn(features)
        return self.classifier(hidden[-1])
