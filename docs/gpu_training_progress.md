# GPU Training Progress

Last updated: 2026-08-21 (Asia/Taipei)

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
competition folds. The frozen aggregation rule is the ceiling of their
arithmetic-mean terminal epoch: `ceil(171 / 15) = ceil(11.4) = 12`. The median
of 11 is descriptive only. The exact per-fold values and hashes of the three
source logs are recorded in `docs/competition_epoch_evidence.md`.

## Pre-run GPU smoke test

On 2026-08-21, B ran the requested single-rotation smoke test on the RTX 3090
before replacing any committed bundle:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  conda run -n aicup-esg --no-capture-output \
  python -u -m paper.run_training \
  --protocol pdf_group --seed 42 --rotations 0 --out-dir /tmp/smoke
```

The real `huggingface_hub.snapshot_download` returned a directory named
`a25cc9e05974bd9687e528edd516f2cfdb3f5db9`, so the new pinned-revision check
accepted the intended snapshot. All 12 epochs completed on the RTX 3090 in
422.0 seconds, and `python -m paper.validate /tmp/smoke/pdf_group_seed42_r0`
reported `1 artifact(s) checked: clean`.

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
minutes per fit, so all 30 official fits completed on the recipe as it stood
then. That recipe has since changed — see **Pending Re-run** at the end of this
file. The table above records what was produced, not what is current.

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
and review follow-up add A's decision-stage, C's analysis and the accumulation
tests. On 2026-08-21 the GPU environment reported `273 passed, 2 skipped`; the
two skips need `latexmk`. A CPU environment without torch skips the training
path as well. The bundle and validation counts are properties of the artifacts
and are unchanged by those optional dependencies.

## Pending Re-run: loss scaling of short batches

**Status: the 30 committed bundles no longer match `paper/train_fold.py`.** They
remain valid, validated artifacts and every number derived from them is
reproducible; what changed is the recipe that produced them.

The losses are `reduction="mean"`, so a batch holding fewer than `BATCH_SIZE`
rows has already inflated each of its rows before any accumulation divisor is
applied. The committed code then divided by the *window's batch count*, which
compounded it. Measured per row, against a normal row's `1/16` of an optimiser
step:

| rotation shape | rows in final batch | weight of those rows |
|---|---:|---:|
| 149 batches (odd), n_train 1185 | 1 | **16x** |
| 151 batches (odd), n_train 1202 | 2 | **8x** |
| 150 batches (even), n_train 1198 | 6 | 1.3x |
| 148 batches (even), n_train 1184 | 8 | 1x (unaffected) |

`paper/accumulation.py::loss_scale` now scales each batch by its real row count
against a fixed denominator, so every row carries `1/16` of a step wherever it
lands and a short window simply takes a proportionally smaller step.
`tests/test_accumulation.py` asserts the property directly.

29 of the 30 bundles have a final batch that is not full and are therefore
affected; only `pdf_group_seed42_r4` (n_train 1184, exactly divisible by 8) is
not. Re-run **all six invocations** rather than 29 rotations: the extra fit
costs about seven minutes and leaves the whole study stamped with one commit,
which `python -m paper.run_manifest` otherwise reports as a mixture.

```bash
for seed in 42 123 456; do
  for protocol in pdf_group row_strat; do
    python -m paper.run_training --protocol $protocol --seed $seed
  done
done
```

`EPOCHS = 12` was re-confirmed from the original competition logs before this
re-run. See `docs/competition_epoch_evidence.md` for the per-fold audit and the
explicit aggregation rule.

After the fits, regenerate the decision stage from a clean checkout — six
invocations of `paper.run_decisions`, a few seconds — so `results/` stops
carrying the `-dirty` stamp, then rebuild the tables.
