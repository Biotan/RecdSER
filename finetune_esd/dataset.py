"""
ESD downstream emotion classification dataset (for RecdSER.finetune_esd).

Reads a catalog.txt with columns [audio_path, emotion_tag, speaker_id], filters
by target emotions, maps labels, and builds a stratified train / val split.
Supports single- and multi-GPU (DDP) training via DistributedSampler.
"""

import os
import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)


class ESDDataset(Dataset):
    """
    ESD emotion classification dataset.

    Args:
        samples: [(audio_path, label_idx, speaker_id), ...]
        audio_root: prefix prepended to audio_path
        sample_rate: target sample rate
        max_sample_size: max waveform length (crop long audio to avoid OOM); None = no crop
    """

    def __init__(
        self,
        samples: List[Tuple[str, int, str]],
        audio_root: str,
        sample_rate: int = 16000,
        max_sample_size: int = None,
    ):
        self.samples = samples
        self.audio_root = audio_root
        self.sample_rate = sample_rate
        self.max_sample_size = max_sample_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        audio_path, label_idx, speaker_id = self.samples[index]
        full_path = os.path.join(self.audio_root, audio_path)

        import soundfile as sf
        waveform, sr = sf.read(full_path)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sr != self.sample_rate:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)

        if self.max_sample_size is not None and waveform.size > self.max_sample_size:
            start = int(np.random.randint(0, waveform.size - self.max_sample_size))
            waveform = waveform[start: start + self.max_sample_size]

        waveform = torch.from_numpy(np.ascontiguousarray(waveform)).float()

        return {
            "waveform": waveform,
            "label": label_idx,
            "speaker_id": speaker_id,
        }

    @staticmethod
    def collator(samples):
        """Pad variable-length waveforms into a batch."""
        if len(samples) == 0:
            return {}

        waveforms = [s["waveform"] for s in samples]
        labels = torch.tensor([s["label"] for s in samples], dtype=torch.long)
        lengths = [w.size(0) for w in waveforms]
        max_len = max(lengths)

        padded_waveforms = torch.zeros(len(waveforms), max_len)
        padding_mask = torch.ones(len(waveforms), max_len, dtype=torch.bool)

        for i, (wav, length) in enumerate(zip(waveforms, lengths)):
            padded_waveforms[i, :length] = wav
            padding_mask[i, :length] = False

        return {
            "waveforms": padded_waveforms,
            "padding_mask": padding_mask,
            "labels": labels,
        }


def parse_catalog(
    catalog_path: str,
    target_emotions: List[str],
    emotion_mapping: Dict[str, str],
    class_labels: List[str],
) -> List[Tuple[str, int, str]]:
    """
    Parse a catalog.txt file into [(audio_path, label_idx, speaker_id), ...].

    Only samples whose (mapped) emotion is in class_labels are kept.
    """
    samples = []
    label_to_idx = {label: idx for idx, label in enumerate(class_labels)}
    skipped = 0

    with open(catalog_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        audio_col = header.index("audio_path")
        emotion_col = header.index("emotion_tag")
        speaker_col = header.index("speaker_id")

        for line in f:
            parts = line.strip().split(",")
            if len(parts) <= max(audio_col, emotion_col, speaker_col):
                skipped += 1
                continue

            audio_path = parts[audio_col]
            emotion_tag = parts[emotion_col]
            speaker_id = parts[speaker_col]

            if emotion_tag not in target_emotions:
                skipped += 1
                continue

            mapped_emotion = emotion_mapping.get(emotion_tag, emotion_tag)
            if mapped_emotion not in label_to_idx:
                skipped += 1
                continue

            label_idx = label_to_idx[mapped_emotion]
            samples.append((audio_path, label_idx, speaker_id))

    logger.info(f"Loaded {len(samples)} samples from catalog, skipped {skipped}")
    return samples


def split_train_val(
    samples: List[Tuple[str, int, str]],
    val_ratio: float,
    seed: int = 42,
) -> Tuple[List[Tuple[str, int, str]], List[Tuple[str, int, str]]]:
    """
    Stratified random split into (train_samples, val_samples).

    - val_ratio == 0.0 : val_samples == [], train_samples == samples (all data for training).
    - 0 < val_ratio < 1 : per-label random split so that class ratios are preserved.

    Returns:
        (train_samples, val_samples)
    """
    if val_ratio < 0.0 or val_ratio >= 1.0:
        raise ValueError(f"val_ratio must be in [0.0, 1.0), got {val_ratio}")

    if val_ratio == 0.0:
        logger.info(f"split_train_val: val_ratio=0 -> all {len(samples)} samples used for training, no val set.")
        return list(samples), []

    rng = np.random.RandomState(seed)
    label_to_indices: Dict[int, List[int]] = {}
    for idx, s in enumerate(samples):
        label_to_indices.setdefault(s[1], []).append(idx)

    train_indices: List[int] = []
    val_indices: List[int] = []
    for label, idx_list in label_to_indices.items():
        arr = np.array(idx_list)
        rng.shuffle(arr)
        n_val = int(round(len(arr) * val_ratio))
        # Guarantee at least 1 in train when class is non-empty.
        n_val = min(n_val, max(len(arr) - 1, 0))
        val_indices.extend(arr[:n_val].tolist())
        train_indices.extend(arr[n_val:].tolist())

    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]

    labels_sorted = sorted(label_to_indices.keys())
    def _dist(lst):
        d: Dict[int, int] = {}
        for s in lst:
            d[s[1]] = d.get(s[1], 0) + 1
        return [d.get(l, 0) for l in labels_sorted]

    logger.info(
        f"split_train_val (val_ratio={val_ratio}): "
        f"train={len(train_samples)}{_dist(train_samples)}, "
        f"val={len(val_samples)}{_dist(val_samples)}, labels={labels_sorted}"
    )
    return train_samples, val_samples


def build_dataloaders(
    train_samples: List,
    val_samples: List,
    test_samples: List,
    audio_root: str,
    sample_rate: int = 16000,
    batch_size: int = 16,
    num_workers: int = 4,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    max_sample_size: int = None,
):
    """Build train/val/test DataLoaders (with DistributedSampler when distributed).

    ``val_samples`` and ``test_samples`` may be ``None`` or an empty list; in that
    case the corresponding loader in the returned tuple will be ``None``.
    """
    train_dataset = ESDDataset(train_samples, audio_root, sample_rate, max_sample_size=max_sample_size)
    val_dataset = ESDDataset(val_samples, audio_root, sample_rate, max_sample_size=max_sample_size) \
        if val_samples else None
    test_dataset = ESDDataset(test_samples, audio_root, sample_rate, max_sample_size=max_sample_size) \
        if test_samples else None

    if distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) \
            if val_dataset is not None else None
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) \
            if test_dataset is not None else None
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=num_workers, pin_memory=True,
        collate_fn=ESDDataset.collator, drop_last=False,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            sampler=val_sampler, num_workers=num_workers, pin_memory=True,
            collate_fn=ESDDataset.collator,
        )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            sampler=test_sampler, num_workers=num_workers, pin_memory=True,
            collate_fn=ESDDataset.collator,
        )

    return train_loader, val_loader, test_loader
