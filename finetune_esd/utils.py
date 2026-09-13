"""
Training & evaluation utilities for RecdSER.finetune_esd.

Metrics:
  - WA  (Weighted Accuracy): overall accuracy
  - UA  (Unweighted Accuracy): macro-average of per-class recall
  - WF1 (Weighted F1): sample-weighted F1
"""

import torch
import torch.distributed as dist
import logging

logger = logging.getLogger(__name__)


def is_dist_initialized():
    return dist.is_available() and dist.is_initialized()


def train_one_epoch(model, optimizer, criterion, train_loader, device, epoch=0,
                    log_interval=10, is_main_process=True, gradient_accumulation_steps=1):
    """Train one epoch (with gradient accumulation). Returns avg loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    total_steps = len(train_loader)

    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        waveforms = batch["waveforms"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(waveforms, padding_mask)
        loss = criterion(logits, labels)
        loss = loss / gradient_accumulation_steps
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == total_steps:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1

        if is_main_process and (step + 1) % log_interval == 0:
            avg_loss_so_far = total_loss / num_batches
            logger.info(
                f"    Epoch {epoch+1} | Step {step+1}/{total_steps} | "
                f"Loss: {loss.item() * gradient_accumulation_steps:.4f} (avg: {avg_loss_so_far:.4f})"
            )

    avg_loss = total_loss / max(num_batches, 1)
    if is_dist_initialized():
        loss_tensor = torch.tensor([avg_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()

    return avg_loss


@torch.no_grad()
def evaluate(model, data_loader, device, num_classes, class_labels=None):
    """
    Evaluate: compute WA, UA, WF1 and per-class P/R/F1.

    Under DDP, each rank evaluates its shard and statistics are aggregated via
    all_reduce before computing metrics.
    """
    model.eval()
    correct, total = 0, 0

    per_class_correct = [0] * num_classes
    per_class_total = [0] * num_classes
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    for batch in data_loader:
        waveforms = batch["waveforms"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(waveforms, padding_mask)
        _, predicted = torch.max(logits, dim=1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for i in range(labels.size(0)):
            gt = labels[i].item()
            pred = predicted[i].item()
            per_class_total[gt] += 1
            if pred == gt:
                per_class_correct[gt] += 1
                tp[gt] += 1
            else:
                fp[pred] += 1
                fn[gt] += 1

    if is_dist_initialized():
        stats = torch.tensor(
            [correct, total] + per_class_correct + per_class_total + tp + fp + fn,
            dtype=torch.long, device=device
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        stats = stats.cpu().tolist()

        correct = stats[0]
        total = stats[1]
        offset = 2
        per_class_correct = stats[offset:offset + num_classes]
        offset += num_classes
        per_class_total = stats[offset:offset + num_classes]
        offset += num_classes
        tp = stats[offset:offset + num_classes]
        offset += num_classes
        fp = stats[offset:offset + num_classes]
        offset += num_classes
        fn = stats[offset:offset + num_classes]

    wa = correct / total * 100 if total > 0 else 0.0
    ua = _compute_ua(per_class_correct, per_class_total) * 100
    wf1 = _compute_wf1(tp, fp, fn, per_class_total) * 100

    per_class_metrics = []
    for i in range(num_classes):
        precision = tp[i] / (tp[i] + fp[i]) * 100 if (tp[i] + fp[i]) > 0 else 0.0
        recall = tp[i] / (tp[i] + fn[i]) * 100 if (tp[i] + fn[i]) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        label_name = class_labels[i] if class_labels and i < len(class_labels) else f"Class_{i}"
        per_class_metrics.append({
            "label": label_name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": per_class_total[i],
        })

    return {"WA": wa, "UA": ua, "WF1": wf1, "per_class": per_class_metrics}


def _compute_ua(per_class_correct, per_class_total):
    recalls = []
    for c, t in zip(per_class_correct, per_class_total):
        if t > 0:
            recalls.append(c / t)
    return sum(recalls) / len(recalls) if recalls else 0.0


def _compute_wf1(tp, fp, fn, per_class_total):
    num_classes = len(tp)
    f1_scores = []
    for i in range(num_classes):
        precision = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
        recall = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
        f1_scores.append(2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)

    total_samples = sum(per_class_total)
    if total_samples == 0:
        return 0.0
    return sum(f1_scores[i] * per_class_total[i] for i in range(num_classes)) / total_samples
