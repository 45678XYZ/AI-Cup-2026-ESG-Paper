"""Train one rotation and emit raw probabilities for its non-train partitions.

Checkpoint selection consults no evaluation labels; see
``train_config.CHECKPOINT_RULE``.

**What this module deliberately does not do** (interface contract §3): it never
applies hard rules, class biases, projection, or argmax, and it never scores
anything. Producing decisions is A's side of the boundary. If training-time
code applied even the hierarchy hard rules, M0 would silently stop being an
unstructured baseline and no number in the results table would reveal it.
"""

import math
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from paper.dataset import ESGDataset, collate_fn
from paper.labels import EVAL_FIELDS, FIELD_WEIGHTS, LABEL2ID, NUM_LABELS
from paper.model import MultiTaskEncoder
from paper.train_config import (
    BACKBONE_LR,
    BATCH_SIZE,
    CHECKPOINT_LAST_K,
    EPOCHS,
    GRAD_ACCUM_STEPS,
    HEAD_LR,
    LABEL_SMOOTHING,
    LR_SCHEDULE,
    MAX_CLASS_WEIGHT,
    MAX_GRAD_NORM,
    MIN_CLASS_SAMPLES_FOR_UPWEIGHT,
    MODEL_NAME,
    MODEL_REVISION,
    WARMUP_RATIO,
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Seed every RNG the training path touches.

    Seed-to-seed std is the reported uncertainty estimate, so the seed has to
    control the model, not just the split.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _class_weights(train_data, field):
    """sklearn's "balanced" weighting, computed here rather than by sklearn.

    ``compute_class_weight("balanced", classes=..., y=...)`` raises when the
    class list contains a label absent from ``y``, and that is not hypothetical:
    ``Misleading`` has 2 rows in the whole dataset, so some pdf_group rotations
    hold both of them out and their Train partition contains none. Calling it
    would abort the rotation before the first step.

    The formula below is sklearn's own -- ``n_samples / (n_present * count)`` --
    so every rotation where all classes are present gets exactly the weights the
    competition recipe produced. An absent class keeps weight 1.0, which the
    minimum-support rule would have assigned it anyway.
    """
    indices = [LABEL2ID[field][row[field]] for row in train_data]
    n_classes = len(EVAL_FIELDS[field])
    counts = np.bincount(indices, minlength=n_classes)

    weights = np.ones(n_classes, dtype=float)
    present = counts > 0
    weights[present] = len(indices) / (present.sum() * counts[present])
    weights = np.clip(weights, 0.1, MAX_CLASS_WEIGHT)
    # Classes with almost no support cannot be learned; upweighting them only
    # injects noise. Absent classes (count 0) fall under this too.
    weights[counts < MIN_CLASS_SAMPLES_FOR_UPWEIGHT] = 1.0
    return torch.tensor(weights, dtype=torch.float, device=DEVICE)


def _loss(criteria, logits, labels):
    total = torch.zeros((), device=DEVICE)
    for field in EVAL_FIELDS:
        total = total + FIELD_WEIGHTS[field] * criteria[field](logits[field], labels[field])
    return total


def _accumulation_window(step, n_batches):
    """Return this window's size and whether ``step`` closes it.

    The final window may contain fewer than ``GRAD_ACCUM_STEPS`` batches. It
    still needs an optimiser step, with its loss scaled by its actual size.
    """
    start = (step // GRAD_ACCUM_STEPS) * GRAD_ACCUM_STEPS
    size = min(GRAD_ACCUM_STEPS, n_batches - start)
    return size, step + 1 == start + size


def train_rotation(train_data, tokenizer, seed, model_name=None, revision=None):
    """Train the anchor on ``train_data`` only.

    Returns ``(model, avg_state)``: the model already carries the averaged
    weights; ``avg_state`` is handed back so the driver can archive it without
    pulling the parameters off the GPU a second time.
    """
    set_seed(seed)

    loader = DataLoader(
        ESGDataset(train_data, tokenizer),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4,
    )
    model = MultiTaskEncoder(
        model_name or MODEL_NAME, NUM_LABELS, revision=revision or MODEL_REVISION,
    ).to(DEVICE)
    optimizer = AdamW(model.get_optimizer_groups(BACKBONE_LR, HEAD_LR))

    total_steps = math.ceil(len(loader) / GRAD_ACCUM_STEPS) * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    make_schedule = (
        get_cosine_schedule_with_warmup if LR_SCHEDULE == "cosine"
        else get_linear_schedule_with_warmup
    )
    scheduler = make_schedule(optimizer, warmup_steps, total_steps)

    criteria = {
        field: nn.CrossEntropyLoss(
            weight=_class_weights(train_data, field), label_smoothing=LABEL_SMOOTHING,
        )
        for field in EVAL_FIELDS
    }
    scaler = torch.cuda.amp.GradScaler()

    # Fixed epoch budget, average the last K states: no evaluation labels are
    # consulted at any point (train_config.CHECKPOINT_RULE).
    recent_states = []

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = {f: batch["labels"][f].to(DEVICE) for f in EVAL_FIELDS}

            with torch.cuda.amp.autocast():
                logits = model(input_ids, attention_mask)
                window_size, update_due = _accumulation_window(step, len(loader))
                loss = _loss(criteria, logits, labels) / window_size

            scaler.scale(loss).backward()

            if update_due:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad()

        print(f"  epoch {epoch + 1}/{EPOCHS} done", flush=True)
        recent_states.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
        if len(recent_states) > CHECKPOINT_LAST_K:
            recent_states.pop(0)

    avg_state = {
        key: torch.stack([s[key] for s in recent_states]).float().mean(0)
        for key in recent_states[0]
    }
    model.load_state_dict({k: v.to(DEVICE) for k, v in avg_state.items()})
    return model, avg_state


@torch.no_grad()
def predict_probs(model, data, tokenizer):
    """Raw softmax probabilities, in the row order of ``data``.

    Returns ``{field: np.ndarray (N, C) float32}`` with columns ordered by
    ``EVAL_FIELDS[field]`` (interface contract §3). No decision rule is applied.
    """
    loader = DataLoader(
        ESGDataset(data, tokenizer, with_labels=False),
        batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4,
    )
    model.eval()
    chunks = {field: [] for field in EVAL_FIELDS}
    for batch in loader:
        logits = model(
            batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE),
        )
        for field in EVAL_FIELDS:
            chunks[field].append(torch.softmax(logits[field], dim=-1).float().cpu().numpy())
    return {
        field: np.concatenate(chunks[field], axis=0).astype(np.float32)
        for field in EVAL_FIELDS
    }
