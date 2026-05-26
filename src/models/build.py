from __future__ import annotations

from torch import nn

from src.models.cnn_avg import CNNAvgModel
from src.models.cnn_lstm import CNNLSTMModel
from src.models.fusion import FusionModel
from src.models.keypoint_lstm import KeypointLSTMModel


def build_model(config: dict) -> nn.Module:
    model_name = config["model"]
    num_classes = int(config.get("num_classes", 8))
    pretrained = bool(config.get("pretrained", True))
    freeze_backbone = bool(config.get("freeze_backbone", True))
    if model_name == "cnn_avg":
        return CNNAvgModel(num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
    if model_name == "cnn_lstm":
        return CNNLSTMModel(num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
    if model_name == "keypoint_lstm":
        return KeypointLSTMModel(num_classes, keypoint_dim=int(config.get("keypoint_dim", 34)))
    if model_name == "fusion":
        return FusionModel(
            num_classes,
            keypoint_dim=int(config.get("keypoint_dim", 34)),
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    raise ValueError(f"Unknown model: {model_name}")
