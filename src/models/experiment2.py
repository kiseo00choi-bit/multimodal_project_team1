from __future__ import annotations

import torch
from torch import nn

from src.models.cnn_avg import resnet18_encoder


class ImageKeypointEstimator(nn.Module):
    """Predicts 17 joint coordinates and visibility from RGB frames."""

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.encoder, feature_dim = resnet18_encoder(pretrained=pretrained)
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 17 * 3),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.encoder.eval()
        return self

    def forward_raw(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, steps, channels, height, width = frames.shape
        x = frames.reshape(bsz * steps, channels, height, width)
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.encoder(x)
        else:
            features = self.encoder(x)
        raw = self.head(features).reshape(bsz, steps, 17, 3)
        xy = torch.sigmoid(raw[..., :2])
        visibility_logits = raw[..., 2]
        return xy, visibility_logits

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        xy, visibility_logits = self.forward_raw(frames)
        visibility = torch.sigmoid(visibility_logits).unsqueeze(-1)
        return (xy * visibility).reshape(frames.shape[0], frames.shape[1], 34)


class KeypointSequenceClassifier(nn.Module):
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

    def forward(self, keypoints: torch.Tensor) -> torch.Tensor:
        x = self.temporal(keypoints.transpose(1, 2)).transpose(1, 2)
        _, hidden = self.rnn(x)
        return self.classifier(hidden[-1])


class RGBSequenceClassifier(nn.Module):
    """RGB-only baseline for experiment2."""

    def __init__(
        self,
        num_classes: int = 8,
        video_hidden_dim: int = 256,
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
        self.video_rnn = nn.GRU(feature_dim, video_hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(video_hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.encoder.eval()
        return self

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        bsz, steps, channels, height, width = frames.shape
        x = frames.reshape(bsz * steps, channels, height, width)
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.encoder(x).reshape(bsz, steps, -1)
        else:
            features = self.encoder(x).reshape(bsz, steps, -1)
        features = self.feature_norm(features)
        _, hidden = self.video_rnn(features)
        return self.classifier(hidden[-1])


class RGBPredictedKeypointFusionClassifier(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        keypoint_dim: int = 34,
        video_hidden_dim: int = 256,
        pose_hidden_dim: int = 128,
        num_classes: int = 8,
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

    def forward(self, frames: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
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
