# GPU Training Progress

Last updated: 2026-08-16 (Asia/Taipei)

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

There are 30 complete probability bundles, each containing `meta.json` and
eight `.npy` arrays. All bundles use the frozen training code and configuration
above. Later bundles may carry an artifact-only checkpoint commit as their Git
SHA; their `train_config_sha256` and training-code contents are unchanged.

| Protocol | Seed | Complete rotations | Status |
|---|---:|---|---|
| `pdf_group` | 42 | r0-r4 | validated clean |
| `pdf_group` | 123 | r0-r4 | validated clean |
| `pdf_group` | 456 | r0-r4 | validated clean |
| `row_strat` | 42 | r0-r4 | validated clean |
| `row_strat` | 123 | r0-r4 | validated clean |
| `row_strat` | 456 | r0-r4 | validated clean |

The final five `row_strat seed456` rotations completed on 2026-08-16 in 6.9-7.0
minutes per fit. All 30 official fits are now complete; no GPU training remains.

Training logs are under `logs/` and intentionally ignored by Git. The completed
bundles carry the exact CLI, Python/Torch/Transformers/CUDA versions, GPU name,
training duration, split fingerprint, Git SHA and array checksums in `meta.json`.

## Final Verification

Run from the repository root:

```bash
conda run -n aicup-esg python -m paper.validate --all
find probs -mindepth 2 -maxdepth 2 -name meta.json | wc -l
conda run -n aicup-esg pytest -q
```

Final observed output on 2026-08-16, on this branch before the merge:

- bundle count: `30`
- `row_strat seed456`: `5 artifact(s) checked: clean`
- all bundles: `30 artifact(s) checked: clean`
- test suite: `92 passed, 2 warnings`

The two warnings are SWIG deprecation warnings emitted during interpreter
shutdown in the `aicup-esg` environment; they do not affect validation or
training artifacts.

The suite count is the one number here that does not carry forward: the merge
brings in A's decision-stage and C's analysis tests, so the same command now
reports `248 passed, 3 skipped`. The three skips are this branch's
`tests/test_training_path.py`, which needs torch, and the two figure tests,
which need `latexmk`. The bundle and validation counts are properties of the
artifacts and are unchanged by the merge.
