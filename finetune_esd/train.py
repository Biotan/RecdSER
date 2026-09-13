"""
Simplified ESD downstream emotion classification training (RecdSER.finetune_esd).

Fine-tunes the Swin-FFA-Net backbone (RecdSER.modules) with an MLP head as
a single end-to-end run (no stage split, backbone is trainable from the start).

Data split (controlled by `dataset.val_ratio` in config.yaml):
  - val_ratio == 0.0       -> ALL data for training, no val set.
  - 0 < val_ratio < 1.0    -> stratified train/val split.
Metrics: WA, UA, WF1. Supports single- and multi-GPU (DDP).

Usage:
  single GPU : python -m RecdSER.finetune_esd.train --config RecdSER/finetune_esd/config.yaml
  multi GPU  : torchrun --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
                 -m RecdSER.finetune_esd.train --config RecdSER/finetune_esd/config.yaml
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path

# Make the project root importable so that `import RecdSER...` works when run directly.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from safetensors.torch import load_file as st_load_file, save_file as st_save_file

from RecdSER.modules import build_swin_ffa_net, EmotionClassifier
from RecdSER.finetune_esd.dataset import (
    parse_catalog, split_train_val, build_dataloaders,
)
from RecdSER.finetune_esd.utils import train_one_epoch, evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("RecdSER.finetune_esd")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def derive_class_labels(target_emotions: list, emotion_mapping: dict) -> list:
    """Derive the final ordered class_labels from target_emotions + emotion_mapping.

    Rules:
      - emotion_mapping empty  -> class_labels = target_emotions (unchanged)
      - emotion_mapping non-empty -> class_labels = order-preserving unique
            values of [emotion_mapping.get(e, e) for e in target_emotions]

    The order of first appearance in target_emotions determines the class index.
    """
    emotion_mapping = emotion_mapping or {}
    if not emotion_mapping:
        return list(target_emotions)
    seen = set()
    ordered = []
    for e in target_emotions:
        mapped = emotion_mapping.get(e, e)
        if mapped not in seen:
            seen.add(mapped)
            ordered.append(mapped)
    return ordered


# ======================== safetensors I/O ========================

def _strip_prefix(sd: dict, prefix: str) -> dict:
    """Strip a leading prefix from every key of the state_dict (if present)."""
    if not sd:
        return sd
    if all(k.startswith(prefix) for k in sd):
        return {k[len(prefix):]: v for k, v in sd.items()}
    return sd


def load_backbone_weights(backbone_model: nn.Module, checkpoint_path: str) -> None:
    """Load pretrained backbone weights (.safetensors) into ``backbone_model``.

    Accepts both flat state_dicts and keys prefixed with ``backbone.``.
    """
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Backbone checkpoint not found: {checkpoint_path}, using random init.")
        return
    if checkpoint_path.endswith(".safetensors"):
        sd = st_load_file(checkpoint_path)
    else:
        sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "backbone" in sd:
            sd = sd["backbone"]
        elif isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
    sd = _strip_prefix(sd, "backbone.")
    missing, unexpected = backbone_model.load_state_dict(sd, strict=False)
    _ignore = ("relative_position_index", "mel_spec.spectrogram.window", "mel_scale.fb")
    real_missing = [k for k in missing if not any(s in k for s in _ignore)]
    real_unexpected = [k for k in unexpected if not any(s in k for s in _ignore)]
    if real_missing:
        logger.warning(f"Missing keys when loading backbone (first 5): {real_missing[:5]}")
    if real_unexpected:
        logger.warning(f"Unexpected keys when loading backbone (first 5): {real_unexpected[:5]}")
    logger.info(f"Loaded Swin-FFA-Net backbone from {checkpoint_path}")


def save_classifier_safetensors(model: EmotionClassifier, class_labels: list, out_path: str) -> None:
    """Save the full EmotionClassifier (backbone + head) as a .safetensors file
    with ``label_class`` stored in the metadata as a JSON-encoded list.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    state_dict = {k: v.contiguous().cpu() for k, v in model.state_dict().items()}
    metadata = {
        "format": "recdser_emotion_classifier_v1",
        "label_class": json.dumps(list(class_labels), ensure_ascii=False),
        "num_classes": str(len(class_labels)),
        "hidden_dim": str(model.classifier.hidden_dim),
        "input_dim": str(model.classifier.input_dim),
    }
    st_save_file(state_dict, out_path, metadata=metadata)
    logger.info(f"Saved classifier -> {out_path} (label_class={class_labels})")


def parse_args():
    parser = argparse.ArgumentParser(description="ESD emotion classification with Swin-FFA-Net backbone")
    parser.add_argument("--config", type=str, default="RecdSER/finetune_esd/config.yaml")
    return parser.parse_args()


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, True
    return 0, 0, 1, False


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def _build_optimizer(name, params, lr, weight_decay):
    if name == "Adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "AdamW":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif name == "RMSprop":
        return optim.RMSprop(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    return optim.Adam(params, lr=lr, weight_decay=weight_decay)


def _build_scheduler(name, optimizer, num_epochs, sched_cfg):
    if name == "CosineAnnealingLR":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=sched_cfg.get("eta_min", 1e-6)
        )
    elif name == "CyclicLR":
        base_lr = sched_cfg.get("base_lr", optimizer.param_groups[0]["lr"])
        max_lr = sched_cfg.get("max_lr", 1e-3)
        step_size_up = sched_cfg.get("step_size_up", 10)
        return optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=base_lr, max_lr=max_lr,
            step_size_up=step_size_up, mode='triangular2',
            cycle_momentum=isinstance(optimizer, (optim.SGD, optim.RMSprop)),
        )
    return None


def main():
    args = parse_args()
    cfg = load_config(args.config)

    rank, local_rank, world_size, is_distributed = setup_distributed()
    is_main_process = (rank == 0)

    common_cfg = cfg.get("common", {})
    dataset_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})
    sched_cfg = cfg.get("lr_scheduler", {})

    seed = common_cfg.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)

    device = torch.device(f"cuda:{local_rank}") if is_distributed else \
        torch.device(common_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    if is_main_process:
        logger.info(f"Using device: {device}, world_size: {world_size}, distributed: {is_distributed}")

    output_dir = common_cfg.get("output_dir", "./RecdSER/finetune_esd/output/ESD")
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
    if is_distributed:
        dist.barrier()

    if is_main_process:
        log_file = os.path.join(output_dir, "train.log")
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logging.getLogger().addHandler(file_handler)

    # -------- dataset --------
    catalog_path = dataset_cfg["catalog_path"]
    audio_root = dataset_cfg["audio_root"]
    sample_rate = dataset_cfg.get("sample_rate", 16000)
    max_sample_size = dataset_cfg.get("max_sample_size", 320000)
    target_emotions = dataset_cfg.get("target_emotions", ["Angry", "Happy", "Neutral", "Sad", "Surprised"])
    emotion_mapping = dataset_cfg.get("emotion_mapping", {}) or {}
    # class_labels are derived automatically from target_emotions + emotion_mapping;
    # NOT read from config anymore.
    class_labels = derive_class_labels(target_emotions, emotion_mapping)
    val_ratio = float(dataset_cfg.get("val_ratio", 0.0))
    batch_size = dataset_cfg.get("batch_size", 16)
    num_workers = dataset_cfg.get("num_workers", 4)
    num_classes = len(class_labels)

    samples = parse_catalog(catalog_path, target_emotions, emotion_mapping, class_labels)
    if is_main_process:
        logger.info(
            f"target_emotions={target_emotions} | emotion_mapping={emotion_mapping} | "
            f"derived class_labels={class_labels} (num_classes={num_classes})"
        )
        logger.info(f"Total samples after filtering: {len(samples)}, classes: {class_labels}")

    train_samples, val_samples = split_train_val(samples, val_ratio, seed=seed)
    if is_main_process:
        if val_ratio == 0.0:
            logger.info(f"Split mode: TRAIN-ONLY (val_ratio=0.0, all {len(samples)} samples for training).")
        else:
            logger.info(
                f"Split mode: TRAIN/VAL random split "
                f"(val_ratio={val_ratio}, train={len(train_samples)}, val={len(val_samples)})."
            )

    # -------- model config --------
    swin_ffa_checkpoint = model_cfg.get("swin_ffa_checkpoint", "./RecdSER/checkpoints/recdser_base.safetensors")
    swin_ffa_config_path = model_cfg.get("swin_ffa_config", "./RecdSER/modules/config.yaml")
    hidden_dim = model_cfg.get("hidden_dim", 256)
    # input_dim of the head is derived from the backbone forward output dim (proj_dim).
    with open(swin_ffa_config_path, "r") as _f:
        _bb_cfg = yaml.safe_load(_f)
    input_dim = model_cfg.get("input_dim", _bb_cfg.get("proj_dim", 768))

    weight_decay = training_cfg.get("weight_decay", 1e-5)
    monitor_metric = training_cfg.get("monitor_metric", "WA")
    optimizer_name = training_cfg.get("optimizer", "Adam")
    use_class_weight = training_cfg.get("use_class_weight", True)

    num_epochs = training_cfg.get("num_epochs", 30)
    learning_rate = training_cfg.get("lr", 1e-5)
    scheduler_name = sched_cfg.get("name", "CyclicLR")

    writer = None
    if is_main_process:
        tb_log_dir = os.path.join(output_dir, "tb_logs")
        writer = SummaryWriter(log_dir=tb_log_dir)
        logger.info(f"TensorBoard log dir: {tb_log_dir}")

    if is_main_process:
        logger.info(f"Training config: optimizer={optimizer_name}, monitor={monitor_metric}, "
                    f"epochs={num_epochs}, lr={learning_rate}, scheduler={scheduler_name}")

    # -------- backbone --------
    if is_main_process:
        logger.info(f"Loading Swin-FFA-Net backbone from: {swin_ffa_checkpoint}")
    backbone_model = build_swin_ffa_net(_bb_cfg).to(device)
    load_backbone_weights(backbone_model, swin_ffa_checkpoint)
    backbone_model.eval()
    original_backbone_state = {k: v.clone() for k, v in backbone_model.state_dict().items()}

    # -------- training (single run) --------
    if is_main_process:
        logger.info(f"{'='*60}")
        logger.info(f"Run: train={len(train_samples)}, val={len(val_samples)}")
        logger.info(f"{'='*60}")

    torch.cuda.empty_cache()
    backbone_model.load_state_dict(original_backbone_state)

    train_loader, val_loader, _ = build_dataloaders(
        train_samples, val_samples, [],
        audio_root=audio_root, sample_rate=sample_rate, batch_size=batch_size,
        num_workers=num_workers, distributed=is_distributed, rank=rank,
        world_size=world_size, max_sample_size=max_sample_size,
    )
    has_val = val_loader is not None

    model = EmotionClassifier(
        backbone=backbone_model, input_dim=input_dim,
        hidden_dim=hidden_dim, num_classes=num_classes,
    ).to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)
    model_module = model.module if is_distributed else model

    # Best classifier weights, saved as safetensors with label_class metadata.
    best_model_path = os.path.join(output_dir, "best_model.safetensors")
    best_metric_value = 0.0
    best_epoch = 0

    # Whole network is trainable from the start; no freeze / unfreeze.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = _build_optimizer(optimizer_name, trainable_params, learning_rate, weight_decay)
    scheduler = _build_scheduler(scheduler_name, optimizer, num_epochs, sched_cfg)

    if use_class_weight:
        from collections import Counter
        import math
        label_counts = Counter([s[1] for s in train_samples])
        class_weights = torch.ones(num_classes, device=device)
        for label_idx, count in label_counts.items():
            class_weights[label_idx] = 1.0 / math.sqrt(count)
        class_weights = class_weights / class_weights.sum() * num_classes
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        if is_distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch + 1)

        train_loss = train_one_epoch(model, optimizer, criterion, train_loader, device,
                                     epoch=epoch, log_interval=10, is_main_process=is_main_process)
        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - epoch_start

        if has_val:
            val_metrics = evaluate(model, val_loader, device, num_classes, class_labels=class_labels)
            current_metric = val_metrics[monitor_metric]

            if is_main_process:
                writer.add_scalar("run/train_loss", train_loss, epoch + 1)
                writer.add_scalar("run/val_WA", val_metrics["WA"], epoch + 1)
                writer.add_scalar("run/val_UA", val_metrics["UA"], epoch + 1)
                writer.add_scalar("run/val_WF1", val_metrics["WF1"], epoch + 1)
                writer.add_scalar("run/lr", optimizer.param_groups[0]["lr"], epoch + 1)

            if current_metric > best_metric_value:
                best_metric_value = current_metric
                best_epoch = epoch + 1
                if is_main_process:
                    save_classifier_safetensors(model_module, class_labels, best_model_path)

            if is_main_process:
                logger.info(
                    f"  Epoch {epoch+1}/{num_epochs} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e} | Loss: {train_loss:.4f} | "
                    f"Val WA: {val_metrics['WA']:.2f}% UA: {val_metrics['UA']:.2f}% "
                    f"WF1: {val_metrics['WF1']:.2f}% | Best {monitor_metric}: "
                    f"{best_metric_value:.2f}% (ep{best_epoch}) | Time: {epoch_time:.1f}s"
                )
        else:
            # No val set: keep saving the latest weights so we retain the model.
            if is_main_process:
                writer.add_scalar("run/train_loss", train_loss, epoch + 1)
                writer.add_scalar("run/lr", optimizer.param_groups[0]["lr"], epoch + 1)
                save_classifier_safetensors(model_module, class_labels, best_model_path)
                best_epoch = epoch + 1
                logger.info(
                    f"  Epoch {epoch+1}/{num_epochs} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e} | Loss: {train_loss:.4f} | "
                    f"(no val; saved latest weights) | Time: {epoch_time:.1f}s"
                )

    if is_main_process:
        if has_val:
            logger.info(f"Training finished. Best {monitor_metric}: "
                        f"{best_metric_value:.2f}% at epoch {best_epoch}")
        else:
            logger.info(f"Training finished (no val). Latest weights saved at epoch {num_epochs}.")
        logger.info(f"Final weights: {best_model_path}")
        writer.flush()
        writer.close()
        logger.info("Training completed.")

    cleanup_distributed()


if __name__ == "__main__":
    main()
