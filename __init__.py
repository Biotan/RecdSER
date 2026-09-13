"""
RecdSER: open-source release of the Swin-FFA-Net based speech emotion recognition system.

Package layout:
  - modules/          : network definitions only
      * backbone.py   : SwinFFANet backbone
      * head.py       : MLP classification head
      * classifier.py : EmotionClassifier = backbone + head
      * config.yaml   : backbone structure config
  - checkpoints/      : released weights
      * recdser_base.safetensors                (backbone only)
      * ESD/recdser_finetune_base.safetensors   (backbone + head, with `label_class` metadata)
  - finetune_esd/     : ESD training only (imports modules/classifier.py)
      trained safetensors -> checkpoints/ESD/ with `label_class` metadata.
  - inference.py      : single inference entry supporting any combination of
                        --output_label / --output_probs / --output_embedding.
"""

from . import modules
from . import finetune_esd

__all__ = ["modules", "finetune_esd"]
