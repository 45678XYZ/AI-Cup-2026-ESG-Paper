# Competition-log evidence for the fixed epoch budget

The controlled study has no labelled validation partition available for early
stopping, so its epoch budget was frozen before the re-run from the prior
competition's 15 standard-loss Chinese RoBERTa folds. This audit resolves what
"typical stopping epoch, rounded up" means: take the arithmetic mean of all 15
terminal epochs, then apply the ceiling.

## Source logs

The source files are intentionally untracked `*.log` files in the original
competition working tree, under `artifacts/`:

| File | SHA-256 |
|---|---|
| `oof_run_roberta_s42.log` | `a97d6b633b038f58f9a3ca543db7bdb88e0d4a611bab5023a7def9b7c4455f0d` |
| `oof_run_roberta_s123.log` | `12b8dcafb812327215c79e9d9103f5e5665e024c44a61bd1788f89da70837c45` |
| `oof_run_roberta_s456.log` | `1dd03a6f696dcaa1092d8cce724b8f7bcca1da13ef9e93e931bbff9ef63e55b4` |

Each log identifies `hfl/chinese-roberta-wwm-ext-large` in its header. Its
epoch records show a 15-epoch cap. For each fold, the terminal epoch is read
from `Early stop at epoch N`; seed 42 fold 3 has no early-stop line and reaches
`Epoch 15/15`, so its terminal epoch is 15.

## Per-fold audit

| Seed | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---:|---:|---:|---:|---:|---:|
| 42 | 12 | 10 | 10 | 15 | 9 |
| 123 | 14 | 14 | 10 | 6 | 14 |
| 456 | 12 | 11 | 11 | 14 | 9 |

The values sum to 171. Therefore:

```text
ceil(arithmetic mean) = ceil(171 / 15) = ceil(11.4) = 12
```

The median is 11, but it is reported only as a descriptive statistic and is
not used by the frozen aggregation rule. Consequently, the audit retains
`EPOCHS = 12` in `paper/train_config.py`.
