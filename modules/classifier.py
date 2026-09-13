"""
RecdSER emotion classifier: composes the Swin-FFA-Net backbone (backbone.py)
with the MLP classification head (head.py).

Contains ONLY the composed network definition. No loader, no training code.
Weight I/O helpers live in ``RecdSER.inference``.
"""

import torch
import torch.nn as nn

from .backbone import SwinFFANet, build_swin_ffa_net
from .head import ClassificationHead


class EmotionClassifier(nn.Module):
    """
    Swin-FFA-Net backbone + MLP classification head.

    Parameter layout (state_dict keys):
      - ``backbone.*``   : Swin-FFA-Net backbone parameters
      - ``classifier.*`` : MLP head parameters (e.g. ``classifier.mlp.0.weight``)
    """

    def __init__(
        self,
        backbone: SwinFFANet,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 5,
    ):
        super().__init__()
        self.backbone = backbone
        self.classifier = ClassificationHead(
            input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes,
        )
        self._backbone_frozen = True
        self.freeze_backbone()

    def forward(self, source: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        if self._backbone_frozen:
            with torch.no_grad():
                embedding = self.backbone(source, padding_mask=padding_mask)
        else:
            embedding = self.backbone(source, padding_mask=padding_mask)
        return self.classifier(embedding)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self._backbone_frozen = True

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        self._backbone_frozen = False

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "_backbone_frozen", True):
            self.backbone.eval()
        return self


def build_emotion_classifier(cfg: dict, input_dim: int, hidden_dim: int, num_classes: int) -> EmotionClassifier:
    """Instantiate an EmotionClassifier from a backbone config dict + head dims."""
    backbone = build_swin_ffa_net(cfg)
    return EmotionClassifier(
        backbone=backbone,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
    )
