# GPU Training Progress

Last updated: 2026-08-15 (Asia/Taipei)

## Frozen Setup

- Branch: `gpu-training-setup`
- Training-code commit: `1e1913cf7cf3f758184409d627d19e1c4f1800a1`
- Conda environment: `aicup-esg`
- GPU: NVIDIA GeForce RTX 3090, 24 GB
- CUDA selection: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`
- Model: `hfl/chinese-roberta-wwm-ext-large`
- Model revision: `a25cc9e05974bd9687e528edd516f2cfdb3f5db9`
- Epochs: 12
- Checkpoint rule: average the last 3 epoch states
- Data checksum: `sha256:e420b03d99aa8cf3353a0bff3bcea49e382946c2f68585d00dc6d0e2b3669e9a`
- Output: raw Calibration/Test softmax probabilities only; no hard rule, bias,
  projection, argmax, or scoring is applied by the training path.

The 12-epoch budget is supported by the 15 standard-loss Chinese RoBERTa
competition folds. Their terminal epochs have median 11 and mean 11.4. The
evidence and exact values are recorded in `paper/train_config.py`.

## Completed Artifacts

There are 21 complete probability bundles, each containing `meta.json` and
eight `.npy` arrays. All completed bundles use the frozen commit above.

| Protocol | Seed | Complete rotations | Status |
|---|---:|---|---|
| `pdf_group` | 42 | r0-r4 | validated clean |
| `pdf_group` | 123 | r0-r4 | validated clean |
| `pdf_group` | 456 | r0-r4 | validated clean |
| `row_strat` | 42 | r0-r4 | validated clean |
| `row_strat` | 123 | r0 | validated clean |
| `row_strat` | 456 | none | not started |

`row_strat seed123 r1` was interrupted after epoch 6. Artifact writing happens
only after training and both inference partitions complete, so no partial bundle
was produced. Resume from r1; do not attempt to reuse those six epochs.

Training logs are under `logs/` and intentionally ignored by Git. The completed
bundles carry the exact CLI, Python/Torch/Transformers/CUDA versions, GPU name,
training duration, split fingerprint, Git SHA and array checksums in `meta.json`.

## Resume Commands

Run from the repository root. `--skip-existing` preserves all complete bundles
and restarts `row_strat seed123` at r1.

```bash
set -o pipefail
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  conda run -n aicup-esg --no-capture-output \
  python -u -m paper.run_training \
  --protocol row_strat --seed 123 --skip-existing \
  2>&1 | tee -a logs/row_strat_seed123.log

env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  conda run -n aicup-esg --no-capture-output \
  python -u -m paper.run_training \
  --protocol row_strat --seed 456 --skip-existing \
  2>&1 | tee logs/row_strat_seed456.log
```

Nine fits remain: `row_strat seed123 r1-r4` and `row_strat seed456 r0-r4`.
Observed runtime is about 7.1 minutes per fit, so expected remaining GPU time is
about 65 minutes.

## Completion Checks

```bash
conda run -n aicup-esg python -m paper.validate probs/row_strat_seed123_r*
conda run -n aicup-esg python -m paper.validate probs/row_strat_seed456_r*
conda run -n aicup-esg python -m paper.validate --all
find probs -mindepth 2 -maxdepth 2 -name meta.json | wc -l  # must print 30
conda run -n aicup-esg pytest -q
```

Each five-rotation validator must report `5 artifact(s) checked: clean`; the
final bundle count must be 30. Commit the remaining `probs/` files only after
these checks pass.
