# GPU Training Progress

Last updated: 2026-08-21 (Asia/Taipei)

## Frozen Setup

- Branch: `gpu-training-setup`
- Artifact-producing commit: `35dea657eede733ea6c8945f3976a1561cfab80d`
- Conda environment: `aicup-esg`
- GPU: NVIDIA GeForce RTX 3090, 24 GB
- CUDA selection: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`
- Model: `hfl/chinese-roberta-wwm-ext-large`
- Model revision: `a25cc9e05974bd9687e528edd516f2cfdb3f5db9`
- Epochs: 12
- Checkpoint rule: average the last 3 epoch states
- Training-config checksum: `sha256:ebef1c618747e263762c86bddc14f6a6ff5faef1b9f4d6fd10a1b3c926929b4c`
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
eight `.npy` arrays. All were regenerated on 2026-08-21 with the corrected
short-batch loss scaling and the frozen setup above. Every bundle records the
same clean Git SHA and training-config checksum.

| Protocol | Seed | Complete rotations | Status |
|---|---:|---|---|
| `pdf_group` | 42 | r0-r4 | validated clean |
| `pdf_group` | 123 | r0-r4 | validated clean |
| `pdf_group` | 456 | r0-r4 | validated clean |
| `row_strat` | 42 | r0-r4 | validated clean |
| `row_strat` | 123 | r0-r4 | validated clean |
| `row_strat` | 456 | r0-r4 | validated clean |

The completed fits took 415.6-424.5 seconds each (210.4 total GPU minutes) on
the RTX 3090. All six invocations completed without a retry.

Training logs are under `logs/` and intentionally ignored by Git. The completed
bundles carry the exact CLI, Python/Torch/Transformers/CUDA versions, GPU name,
training duration, split fingerprint, Git SHA and array checksums in `meta.json`.

## Final Verification

Run from the repository root:

```bash
conda run -n aicup-esg python -m paper.validate --all
find probs -mindepth 2 -maxdepth 2 -name meta.json | wc -l
conda run -n aicup-esg pytest -q
conda run -n aicup-esg python -m paper.run_manifest
conda run -n aicup-esg python -m analysis
```

Final observed output on 2026-08-21 after rebuilding every downstream artifact:

- bundle count: `30`
- all contract artifacts: `72 artifact(s) checked: clean`
- test suite: `273 passed, 2 skipped, 2 warnings`
- study manifest: 6/6 splits, 30/30 probabilities, 42/42 predictions,
  42/42 results, eight table artifacts and six cross-file checks passed with
  no warnings or notes
- analysis: Table 1-3 rebuilt; Figure 1 was unchanged because `latexmk` is not
  installed in this environment

The two warnings are SWIG deprecation warnings emitted during interpreter
shutdown in the `aicup-esg` environment; they do not affect validation or
training artifacts.

The two test skips need `latexmk`. A CPU environment without torch skips the
training path as well, so its suite count differs while the contract checks do
not.

## Completed re-run: loss scaling of short batches

**Status: fixed and re-run.** The 30 committed bundles now match
`paper/train_fold.py` and carry one clean provenance stamp.

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

29 of the 30 superseded bundles had a final batch that was not full and were
therefore affected; only `pdf_group_seed42_r4` (n_train 1184, exactly divisible
by 8) was not. All six invocations were nevertheless re-run so the entire study
uses one recipe and one commit:

```bash
for seed in 42 123 456; do
  for protocol in pdf_group row_strat; do
    python -m paper.run_training --protocol $protocol --seed $seed
  done
done
```

`EPOCHS = 12` was re-confirmed from the original competition logs before the
campaign. See `docs/competition_epoch_evidence.md` for the per-fold audit and
the explicit aggregation rule. After the fits, all six decision invocations,
the study manifest and Table 1-3 were regenerated from the clean source commit.
