"""Torch dataset and collation.

Unlabelled rows are supported so the same dataset can wrap inference
partitions; ``labels`` is simply absent from the batch in that case.
"""

import torch
from torch.utils.data import Dataset

from paper.labels import EVAL_FIELDS, LABEL2ID
from paper.train_config import MAX_LEN


class ESGDataset(Dataset):
    def __init__(self, data, tokenizer, with_labels=True):
        self.data = data
        self.tokenizer = tokenizer
        self.with_labels = with_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        enc = self.tokenizer(
            sample["data"],
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
        )
        item = {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
        }
        if self.with_labels:
            item["labels"] = {
                field: torch.tensor(LABEL2ID[field][sample[field]], dtype=torch.long)
                for field in EVAL_FIELDS
            }
        return item


def collate_fn(batch):
    out = {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
    }
    if "labels" in batch[0]:
        out["labels"] = {
            field: torch.stack([b["labels"][field] for b in batch])
            for field in EVAL_FIELDS
        }
    return out
