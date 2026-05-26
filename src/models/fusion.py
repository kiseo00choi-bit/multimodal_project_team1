from __future__ import annotations

import torch
from torch import nn

from src.models.cnn_avg import resnet18_encoder


class FusionModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        keypoint_dim: int = 34,
        video_hidden_dim: int = 256,
        pose_hidden_dim: int = 128,
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
        self.video_feature_norm = nn.LayerNorm(feature_dim)
        self.video_rnn = nn.GRU(feature_dim, video_hidden_dim, batch_first=True)
        self.pose_temporal = nn.Sequential(
            nn.Conv1d(keypoint_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.pose_rnn = nn.GRU(128, pose_hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(video_hidden_dim + pose_hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.encoder.eval()
        return self

    def forward(self, frames: torch.Tensor | None = None, keypoints: torch.Tensor | None = None) -> torch.Tensor:
        if frames is None or keypoints is None:
            raise ValueError("frames and keypoints are required")
        bsz, steps, channels, height, width = frames.shape
        x = frames.reshape(bsz * steps, channels, height, width)
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.encoder(x).reshape(bsz, steps, -1)
        else:
            features = self.encoder(x).reshape(bsz, steps, -1)
        features = self.video_feature_norm(features)
        _, video_hidden = self.video_rnn(features)

        pose = self.pose_temporal(keypoints.transpose(1, 2)).transpose(1, 2)
        _, pose_hidden = self.pose_rnn(pose)

        fused = torch.cat([video_hidden[-1], pose_hidden[-1]], dim=-1)
        return self.classifier(fused)
