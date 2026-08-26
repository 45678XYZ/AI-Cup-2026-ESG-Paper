# Documents

Sorted by how they are used, not by topic.

## `governance/` — constrains what may be done

Editing these has consequences beyond the file.

| | |
|---|---|
| `paper_plan.md` | The frozen analysis plan. **Never edited after results were seen** -- the pre-specification argument only holds if that is verifiable in the history. |
| `interface_contract.md` | What A, B, C and D deliver to each other, and in what shape. Changing a declared shape is a contract change. |
| `inference_families.md` | Which test belongs to which family, and what each layer may claim. Settled 2026-08-24. |
| `competition_epoch_evidence.md` | Where `EPOCHS = 12` came from. Cited by `paper/train_config.py` as the justification for a frozen constant. |

`paper/train_config.py` cites this file at its pre-move path, `docs/competition_epoch_evidence.md`, and that stale path is deliberate. The whole file is pinned by `train_config_sha256`, which 166 versioned artifacts record; a comment is hashed exactly like a constant, so correcting the path would invalidate every completed official fit. `tests/test_train_config_frozen.py` fails if anyone tries.

## `preregistration/` — written before the run

Each was committed before its arm executed; the commit timestamp is the evidence.
**Not edited afterwards** -- results go in `results/`.

| | Arm |
|---|---|
| `pre_registration_structural_training.md` | RoBERTa-large, structural loss |
| `pre_registration_deberta_screen.md` | DeBERTa-v2-320M |
| `pre_registration_electra_screen.md` | ELECTRA-large |
| `rbt_base_run.md` | Chinese RoBERTa-base |
| `pre_registration_english_replication.md` | ML-Promise English, the external replication |
| `pre_registration_multilingual_replication.md` | ML-Promise French, Japanese and Korean |

## `results/` — written after the run

One per arm, plus the indexes.

| | |
|---|---|
| `structural_training_results.md` | |
| `deberta_screen_results.md` | |
| `electra_screen_results.md` | |
| `rbt_base_results.md` | |
| `multilingual_replication_results.md` | The French/Japanese/Korean campaign: what ran, and the per-language numbers |
| `multilingual_mechanism_results.md` | The frozen study's mechanism re-measured on all five corpora, and the finding that the metric's leverage is per-corpus |
| `consolidated_runs_index.md` | What the four execution branches produced, and how they were merged |
| `gpu_training_progress.md` | B's training log |

## `writing/` — for the paper

| | |
|---|---|
| `study_report.md` | **Start here.** The claim, the evidence chain, how to write each section, the red lines, and the reviewer questions. |
| `related_work_citations.md` | The path-constrained metric traced to its sources, with BibTeX and verbatim quotes. ⚠️ Covers the metric's lineage only, not the whole Related Work; its section on the tension with Plaud et al. still assumes that metric is a pre-specified family, which it stopped being when the analysis demoted it to exploratory. |

---

Numbers in any of these are secondary to the generated deliverables. When they
disagree, `tables/findings.md` wins.
