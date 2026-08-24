# Experiment runs

Everything here is an arm that was added after the frozen study. One directory
per training configuration, all in the same shape:

```
<arm>/
  probs/          probability bundles, one per (protocol, seed, rotation)
  predictions/    per-row decisions, M0-M6
  results/        per-run summaries
  comparison.json the arm's own analysis
```

The frozen anchor is **not** here. It lives at the repository root -- `probs/`,
`predictions/`, `results/` -- because contract section 4 names those paths and
`run_manifest.json`, `paper/validate.py --all` and every table's manifest are
built against them. Moving it would break the audit trail it exists to provide.

| Directory | Backbone | $\lambda$ | Registered in |
|---|---|---|---|
| `structural/` | RoBERTa-large | 0.3 | `docs/pre_registration_structural_training.md` |
| `lambda_sweep/` | RoBERTa-large | 0.1 / 0.3 / 1.0 | same, §7 -- selection only, scored into no table |
| `deberta_v2_320m/` | DeBERTa-v2-320M | 0.0 and 0.3 | `docs/pre_registration_deberta_screen.md` |
| `electra_180g_large/` | ELECTRA-large | 0.0 and 0.3 | `docs/pre_registration_electra_screen.md` |
| `rbt_base/` | Chinese RoBERTa-base | 0.0 | `docs/rbt_base_run.md` |

The two architecture screens carry a `lambda_0.0/` and a `lambda_0.3/` because
each ran both arms itself; the others were compared against the frozen anchor.

`lambda_sweep/` holds probability bundles only. It selected $\lambda$ on the
Calibration partition and its pre-registration forbids it from entering any
result table -- keeping it here rather than deleting it is what makes that
selection checkable afterwards.

Which arm feeds which table: `analysis/legality_cost.py::ARMS`.
