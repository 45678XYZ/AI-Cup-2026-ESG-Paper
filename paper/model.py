"""Shared-encoder, four-head classifier.

A plain shared encoder with four independent linear heads. The study is about
*decision-stage* structure, not architecture, so the encoder carries no
hierarchical head conditioning and no multi-layer pooling.

``AutoModel`` rather than a hard-coded ``BertModel``: the frozen anchor is a
BERT-architecture checkpoint, but nothing here depends on that.
"""

import torch
import torch.nn as nn
from transformers import AutoModel

from paper.train_config import (
    DROPOUT,
    LLRD_DECAY,
    NO_DECAY,
    USE_LLRD,
    USE_MEAN_POOLING,
    WEIGHT_DECAY,
)


class MultiTaskEncoder(nn.Module):
    def __init__(self, model_name, num_labels_dict, dropout=DROPOUT, revision=None,
                 local_files_only=False):
        super().__init__()
        # ``revision`` is honoured only when ``model_name`` is a Hub id; for a
        # local snapshot directory transformers ignores it, so the driver pins
        # the revision by resolving the snapshot itself (run_training.py).
        self.encoder = AutoModel.from_pretrained(
            model_name, revision=revision, local_files_only=local_files_only,
        )
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifiers = nn.ModuleDict({
            field: nn.Linear(hidden_size, n) for field, n in num_labels_dict.items()
        })

    def _pool(self, out, attention_mask):
        if USE_MEAN_POOLING:
            h = out.last_hidden_state
            mask = attention_mask.unsqueeze(-1).float()
            return (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return out.pooler_output

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(self._pool(out, attention_mask))
        return {field: clf(pooled) for field, clf in self.classifiers.items()}

    @staticmethod
    def _decay_groups(named_params, lr):
        """One learning rate, split into decayed and undecayed parameters.

        Every group carries an explicit ``weight_decay`` so the optimiser can
        never fall back on a library default that ``train_config`` does not
        record (see ``WEIGHT_DECAY`` there).
        """
        decay, no_decay = [], []
        for name, param in named_params:
            (no_decay if any(k in name for k in NO_DECAY) else decay).append(param)
        groups = []
        if decay:
            groups.append({"params": decay, "lr": lr, "weight_decay": WEIGHT_DECAY})
        if no_decay:
            groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
        return groups

    def get_optimizer_groups(self, backbone_lr, head_lr):
        """Parameter groups, with layer-wise LR decay when enabled."""
        if not USE_LLRD:
            return (
                self._decay_groups(self.encoder.named_parameters(), backbone_lr)
                + self._decay_groups(self.classifiers.named_parameters(), head_lr)
            )

        groups = self._decay_groups(self.classifiers.named_parameters(), head_lr)

        n_layers = len(self.encoder.encoder.layer)
        groups += self._decay_groups(
            self.encoder.embeddings.named_parameters(),
            backbone_lr * (LLRD_DECAY ** n_layers),
        )
        for i, layer in enumerate(self.encoder.encoder.layer):
            groups += self._decay_groups(
                layer.named_parameters(),
                backbone_lr * (LLRD_DECAY ** (n_layers - 1 - i)),
            )
        if getattr(self.encoder, "pooler", None) is not None:
            groups += self._decay_groups(self.encoder.pooler.named_parameters(), backbone_lr)

        return groups
