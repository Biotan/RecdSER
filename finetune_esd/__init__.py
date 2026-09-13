"""
RecdSER.finetune_esd: simplified training for the ESD downstream task.

This subpackage ONLY handles training. Network definitions come from
``RecdSER.modules`` (backbone.py / head.py / classifier.py). Inference lives
in the top-level ``RecdSER.inference`` module.

Trained safetensors are written to ``RecdSER/checkpoints/ESD/`` and carry a
``label_class`` metadata field so downstream inference can recover the
per-dimension label names of that dataset.
"""

from .dataset import ESDDataset, parse_catalog, split_train_val, build_dataloaders
from .utils import train_one_epoch, evaluate

__all__ = [
    "ESDDataset",
    "parse_catalog",
    "split_train_val",
    "build_dataloaders",
    "train_one_epoch",
    "evaluate",
]
