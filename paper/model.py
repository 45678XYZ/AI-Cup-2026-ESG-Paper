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

from paper.train_config import DROPOUT, LLRD_DECAY, USE_LLRD, USE_MEAN_POOLING


class MultiTaskEncoder(nn.Module):
    def __init__(self, model_name, num_labels_dict, dropout=DROPOUT, revision=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, revision=revision)
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

    def get_optimizer_groups(self, backbone_lr, head_lr):
        """Parameter groups, with layer-wise LR decay when enabled."""
        if not USE_LLRD:
            return [
                {"params": list(self.encoder.parameters()), "lr": backbone_lr},
                {"params": list(self.classifiers.parameters()), "lr": head_lr},
            ]

        groups = [{"params": list(self.classifiers.parameters()), "lr": head_lr}]

        n_layers = len(self.encoder.encoder.layer)
        groups.append({
            "params": list(self.encoder.embeddings.parameters()),
            "lr": backbone_lr * (LLRD_DECAY ** n_layers),
        })
        for i, layer in enumerate(self.encoder.encoder.layer):
            groups.append({
                "params": list(layer.parameters()),
                "lr": backbone_lr * (LLRD_DECAY ** (n_layers - 1 - i)),
            })
        if getattr(self.encoder, "pooler", None) is not None:
            groups.append({"params": list(self.encoder.pooler.parameters()), "lr": backbone_lr})

        return groups
