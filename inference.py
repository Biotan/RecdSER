"""
RecdSER inference: single entry point for embedding extraction and/or emotion
classification.

Command-line switches control what is emitted:
    --output_label       : predicted top-1 label (requires a checkpoint with head)
    --output_probs       : per-class softmax probabilities (requires head)
    --output_embedding   : 768-dim L2-normalized utterance embedding
Any combination is allowed; at least one must be enabled.

Checkpoint files
----------------
* ``recdser_base.safetensors`` (backbone-only)   -> supports ONLY --output_embedding
* ``ESD/recdser_finetune_base.safetensors`` and other finetuned files
  (backbone + head, with ``label_class`` metadata) -> supports all three outputs

The set of class labels is auto-loaded from the checkpoint's safetensors
``metadata['label_class']`` (JSON-encoded list). Users can override with
``--class_labels`` (mostly for backwards compatibility).

Typical CLI usage
-----------------
Embedding only::

    python -m RecdSER.inference \
        --checkpoint RecdSER/checkpoints/recdser_base.safetensors \
        --audio a.wav b.wav \
        --output_embedding --output out.npy

Label + probs (from ESD-finetuned checkpoint)::

    python -m RecdSER.inference \
        --checkpoint RecdSER/checkpoints/ESD/recdser_finetune_base.safetensors \
        --audio a.wav \
        --output_label --output_probs
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

# Make the project root importable so `import RecdSER...` works when run directly.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
import yaml
from safetensors import safe_open
from safetensors.torch import load_file as st_load_file

from RecdSER.modules import build_swin_ffa_net, build_emotion_classifier

logger = logging.getLogger("RecdSER.inference")

_DEFAULT_CONFIG = str(Path(__file__).resolve().parent / "modules" / "config.yaml")


# ----------------------------- audio I/O -----------------------------

def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load an audio file as a mono float waveform at ``target_sr``."""
    import soundfile as sf
    try:
        waveform, sr = sf.read(path, always_2d=False)
    except Exception:
        import librosa
        waveform, sr = librosa.load(path, sr=target_sr, mono=True)
        return waveform.astype(np.float32)

    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sr != target_sr:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
    return waveform.astype(np.float32)


def _list_audio_files(paths):
    exts = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac")
    out = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            out.extend(sorted(str(x) for x in pp.rglob("*") if x.suffix.lower() in exts))
        else:
            out.append(str(pp))
    return out


def _waveforms_to_batch(waveforms, device, max_sample_size=None):
    tensors = []
    for wav in waveforms:
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav)
        wav = wav.float()
        if max_sample_size is not None and wav.size(0) > max_sample_size:
            start = int(torch.randint(0, wav.size(0) - max_sample_size, (1,)).item())
            wav = wav[start:start + max_sample_size]
        tensors.append(wav)

    max_len = max(t.size(0) for t in tensors)
    batch = torch.zeros(len(tensors), max_len)
    padding_mask = torch.ones(len(tensors), max_len, dtype=torch.bool)
    for i, t in enumerate(tensors):
        batch[i, :t.size(0)] = t
        padding_mask[i, :t.size(0)] = False
    return batch.to(device), padding_mask.to(device)


# ----------------------------- checkpoint parsing -----------------------------

def read_checkpoint_info(checkpoint_path: str) -> dict:
    """
    Peek into a safetensors checkpoint and return:
        {
          "has_head": bool,
          "label_class": list[str] | None,
          "num_classes": int | None,
          "hidden_dim": int | None,
          "input_dim": int | None,
          "keys": list[str],
          "metadata": dict,
        }
    """
    if not checkpoint_path.endswith(".safetensors"):
        raise ValueError(f"Only .safetensors checkpoints are supported, got: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        metadata = dict(f.metadata() or {})

    has_head = any(k.startswith("classifier.") for k in keys)
    label_class = None
    if "label_class" in metadata:
        try:
            label_class = json.loads(metadata["label_class"])
        except Exception:
            label_class = None

    def _as_int(v):
        try:
            return int(v)
        except Exception:
            return None

    return {
        "has_head": has_head,
        "label_class": label_class,
        "num_classes": _as_int(metadata.get("num_classes")),
        "hidden_dim": _as_int(metadata.get("hidden_dim")),
        "input_dim": _as_int(metadata.get("input_dim")),
        "keys": keys,
        "metadata": metadata,
    }


# ----------------------------- model loading -----------------------------

def _strip_backbone_prefix(sd: dict) -> dict:
    if all(k.startswith("backbone.") for k in sd):
        return {k[len("backbone."):]: v for k, v in sd.items()}
    return sd


def load_model(
    checkpoint_path: str,
    config_path: str,
    device: torch.device,
    *,
    need_head: bool,
    override_class_labels=None,
    override_hidden_dim: int = None,
):
    """
    Load either a bare SwinFFANet (embedding-only) or a full EmotionClassifier
    (backbone + head).

    Returns:
        model         : nn.Module in eval mode on ``device``.
        class_labels  : list[str] or None (None when head is absent).
        info          : dict returned by :func:`read_checkpoint_info`.
    """
    info = read_checkpoint_info(checkpoint_path)
    with open(config_path, "r") as f:
        bb_cfg = yaml.safe_load(f)

    if need_head and not info["has_head"]:
        raise ValueError(
            f"Checkpoint {checkpoint_path} does NOT contain a classification head "
            f"('classifier.*' keys are missing). --output_label / --output_probs "
            f"require a checkpoint with the head (e.g. one produced by "
            f"finetune_esd/train.py). Backbone-only weights can only be used for "
            f"embedding extraction."
        )

    state_dict = st_load_file(checkpoint_path)
    _ignore = ("relative_position_index", "mel_spec.spectrogram.window", "mel_scale.fb")

    if info["has_head"]:
        # Build the composed classifier.
        class_labels = override_class_labels or info["label_class"]
        if class_labels is None:
            raise ValueError(
                f"Checkpoint {checkpoint_path} contains a head but no 'label_class' "
                f"metadata. Pass --class_labels to override, or re-save the "
                f"checkpoint with proper metadata (see finetune_esd/train.py: "
                f"save_classifier_safetensors)."
            )
        hidden_dim = override_hidden_dim or info["hidden_dim"] or 256
        input_dim = info["input_dim"] or bb_cfg.get("proj_dim", 768)
        num_classes = len(class_labels)

        model = build_emotion_classifier(bb_cfg, input_dim=input_dim,
                                          hidden_dim=hidden_dim, num_classes=num_classes)
        ckpt_total_keys = len(state_dict)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        model_arch = "EmotionClassifier (backbone + head)"
    else:
        # Build a bare backbone.
        class_labels = None
        model = build_swin_ffa_net(bb_cfg)
        state_dict = _strip_backbone_prefix(state_dict)
        # Drop any classifier.* leftover (shouldn't happen here, but be robust).
        ckpt_total_keys = len(state_dict)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("classifier.")}
        dropped_classifier = ckpt_total_keys - len(state_dict)
        ckpt_total_keys = len(state_dict)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        model_arch = "SwinFFANet (backbone only)"
        if dropped_classifier:
            logger.info(f"[load] dropped {dropped_classifier} 'classifier.*' key(s) from checkpoint "
                        f"(backbone-only build ignores head weights).")

    # ---------- detailed load report ----------
    ignored_missing = [k for k in missing if any(s in k for s in _ignore)]
    real_missing = [k for k in missing if k not in ignored_missing]
    ignored_unexpected = [k for k in unexpected if any(s in k for s in _ignore)]
    real_unexpected = [k for k in unexpected if k not in ignored_unexpected]

    model_total_keys = len(model.state_dict())
    loaded_keys = model_total_keys - len(missing)
    strict_match = (len(real_missing) == 0 and len(real_unexpected) == 0)

    logger.info(f"[load] checkpoint = {checkpoint_path}")
    logger.info(f"[load] model arch = {model_arch}")
    logger.info(f"[load] model expects {model_total_keys} keys | "
                f"checkpoint provides {ckpt_total_keys} keys | "
                f"successfully loaded {loaded_keys} keys")
    if strict_match:
        logger.info(f"[load] ✅ STRICT MATCH: all model parameters were loaded and no extra "
                    f"parameters remain (ignoring known non-persistent buffers).")
    else:
        logger.warning(f"[load] ⚠ NOT a strict match: "
                       f"{len(real_missing)} missing, {len(real_unexpected)} unexpected.")

    if real_missing:
        logger.warning(f"[load] Missing keys ({len(real_missing)}) — parameters the model "
                       f"expects but were NOT found in the checkpoint:")
        for k in real_missing:
            logger.warning(f"        - {k}")
    if real_unexpected:
        logger.warning(f"[load] Unexpected keys ({len(real_unexpected)}) — extra tensors in "
                       f"the checkpoint that the model does NOT use:")
        for k in real_unexpected:
            logger.warning(f"        + {k}")
    if ignored_missing or ignored_unexpected:
        logger.info(f"[load] (ignored) non-persistent buffers auto-skipped: "
                    f"missing={len(ignored_missing)}, unexpected={len(ignored_unexpected)} "
                    f"(matched patterns: {list(_ignore)}).")

    model = model.to(device)
    model.eval()
    return model, class_labels, info


# ----------------------------- inference primitives -----------------------------

@torch.no_grad()
def run_inference(
    waveforms: list,
    checkpoint: str,
    config: str = None,
    device: str = "cpu",
    output_embedding: bool = False,
    output_label: bool = False,
    output_probs: bool = False,
    class_labels_override=None,
    max_sample_size: int = 320000,
):
    """
    Run inference over a list of waveforms.

    Returns a dict with the requested outputs:
        {
          "class_labels" : list[str] | None,
          "embedding"    : np.ndarray (N, D) if output_embedding else None,
          "label"        : list[str/int]     if output_label     else None,
          "probs"        : np.ndarray (N, C) if output_probs     else None,
        }
    """
    if not (output_embedding or output_label or output_probs):
        raise ValueError("At least one of output_embedding / output_label / output_probs must be True.")

    if config is None:
        config = _DEFAULT_CONFIG

    need_head = output_label or output_probs
    model, class_labels, info = load_model(
        checkpoint, config, torch.device(device),
        need_head=need_head, override_class_labels=class_labels_override,
    )

    batch, padding_mask = _waveforms_to_batch(waveforms, device, max_sample_size)

    result = {"class_labels": class_labels,
              "embedding": None, "label": None, "probs": None}

    # Route by the ACTUAL model that was built (has head or not), not by user intent.
    # When a finetuned checkpoint is loaded, `model` is an EmotionClassifier whose
    # forward() returns logits; the 768-D embedding must be taken from `.backbone`.
    if info["has_head"]:
        # EmotionClassifier: run head only if the user asked for label/probs.
        if need_head:
            logits = model(batch, padding_mask=padding_mask)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
            if output_probs:
                result["probs"] = probs
            if output_label:
                result["label"] = [class_labels[i] for i in preds] if class_labels \
                                  else [int(i) for i in preds]
        if output_embedding:
            emb = model.backbone(batch, padding_mask=padding_mask)
            result["embedding"] = emb.cpu().numpy()
    else:
        # Bare backbone: forward -> embedding.
        emb = model(batch, padding_mask=padding_mask)
        result["embedding"] = emb.cpu().numpy()

    return result


# ----------------------------- CLI -----------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="RecdSER inference: embedding / label / probs (any combination)."
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help=".safetensors weight file under RecdSER/checkpoints/.")
    parser.add_argument("--config", type=str, default=None,
                        help="Backbone config (defaults to RecdSER/modules/config.yaml).")
    parser.add_argument("--audio", type=str, nargs="+", required=True,
                        help="Audio file(s) and/or directorie(s).")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_sample_size", type=int, default=320000,
                        help="Max waveform length in samples (crops longer audio).")

    parser.add_argument("--output_label", action="store_true",
                        help="Output top-1 predicted label (requires head).")
    parser.add_argument("--output_probs", action="store_true",
                        help="Output per-class softmax probabilities (requires head).")
    parser.add_argument("--output_embedding", action="store_true",
                        help="Output 768-dim L2-normalized utterance embedding.")

    parser.add_argument("--class_labels", type=str, nargs="+", default=None,
                        help="Override the label_class metadata stored in the checkpoint.")
    parser.add_argument("--output", type=str, default=None,
                        help="Write results to this path "
                             "(.json for label/probs, .npy for embedding, or .npz if both).")
    return parser.parse_args()


def _format_stdout(item: dict) -> str:
    parts = [item["audio"]]
    if "label" in item:
        parts.append(f"label={item['label']}")
    if "probs" in item:
        parts.append("probs={" + ", ".join(f"{k}:{v:.3f}" for k, v in item["probs"].items()) + "}")
    if "embedding_shape" in item:
        parts.append(f"embedding_shape={item['embedding_shape']}")
    return "\t".join(parts)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = _parse_args()

    if not (args.output_label or args.output_probs or args.output_embedding):
        raise SystemExit("Enable at least one of --output_label / --output_probs / --output_embedding.")

    audio_files = _list_audio_files(args.audio)
    logger.info(f"Found {len(audio_files)} audio file(s).")
    waveforms = [load_audio(p, target_sr=16000) for p in audio_files]

    result = run_inference(
        waveforms, args.checkpoint, args.config, args.device,
        output_embedding=args.output_embedding,
        output_label=args.output_label,
        output_probs=args.output_probs,
        class_labels_override=args.class_labels,
        max_sample_size=args.max_sample_size,
    )
    class_labels = result["class_labels"]

    # Build per-audio records.
    records = []
    for i, p in enumerate(audio_files):
        item = {"audio": p}
        if result["label"] is not None:
            item["label"] = result["label"][i]
        if result["probs"] is not None:
            probs_i = result["probs"][i]
            item["probs"] = {class_labels[j]: float(probs_i[j]) for j in range(len(probs_i))} \
                if class_labels else {str(j): float(probs_i[j]) for j in range(len(probs_i))}
        if result["embedding"] is not None:
            item["embedding_shape"] = list(result["embedding"][i].shape)
        records.append(item)

    # Persist.
    if args.output:
        ext = os.path.splitext(args.output)[1].lower()
        if result["embedding"] is not None and (result["label"] is not None or result["probs"] is not None):
            # Both -> .npz bundle.
            out_path = args.output if ext == ".npz" else args.output + ".npz"
            np.savez(
                out_path,
                audio=np.array(audio_files),
                embedding=result["embedding"],
                probs=result["probs"] if result["probs"] is not None else np.array([]),
                labels=np.array(result["label"]) if result["label"] is not None else np.array([]),
                class_labels=np.array(class_labels) if class_labels else np.array([]),
            )
            logger.info(f"Saved bundle -> {out_path}")
        elif result["embedding"] is not None:
            out_path = args.output if ext == ".npy" else args.output + ".npy"
            np.save(out_path, result["embedding"])
            logger.info(f"Saved embeddings -> {out_path} (shape={result['embedding'].shape}).")
        else:
            out_path = args.output if ext == ".json" else args.output + ".json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"class_labels": class_labels, "results": records}, f,
                          ensure_ascii=False, indent=2)
            logger.info(f"Saved results -> {out_path}")
    else:
        for r in records:
            print(_format_stdout(r))


if __name__ == "__main__":
    main()
