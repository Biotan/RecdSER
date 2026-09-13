"""
RecdSER downstream classification head.

Contains ONLY the MLP classification head network definition. No loader, no
training code. The composed network (backbone + head) is defined in
``classifier.py``.
"""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """MLP classification head."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, num_classes: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
