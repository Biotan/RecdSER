"""
RecdSER.modules: network definitions only (no training / loading code).

  - backbone.py   : SwinFFANet backbone
  - head.py       : MLP classification head
  - classifier.py : EmotionClassifier = backbone + head
"""

from .backbone import SwinFFANet, build_swin_ffa_net
from .head import ClassificationHead
from .classifier import EmotionClassifier, build_emotion_classifier

__all__ = [
    "SwinFFANet",
    "build_swin_ffa_net",
    "ClassificationHead",
    "EmotionClassifier",
    "build_emotion_classifier",
]
