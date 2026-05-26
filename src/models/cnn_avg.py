from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn
from torchvision import models


def resnet18_encoder(pretrained: bool = True) -> tuple[nn.Module, int]:
    os.environ.setdefault("TORCH_HOME", str(Path(".cache/torch").resolve()))
    weights = None
    if pretrained:
        try:
            weights = models.ResNet18_Weights.DEFAULT
        except Exception:
            weights = None
    try:
        backbone = models.resnet18(weights=weights)
    except Exception as exc:
        print(f"Could not load pretrained ResNet18 weights ({exc}); using random init.")
        backbone = models.resnet18(weights=None)
    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, feature_dim


class CNNAvgModel(nn.Module):
    def __init__(self, num_classes: int = 8, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.encoder, feature_dim = resnet18_encoder(pretrained=pretrained)
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, num_classes),
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
        pooled = self.feature_norm(features.mean(dim=1))
        return self.classifier(pooled)
