"""What the intervals license D to write, computed rather than agreed verbally.

C is the only person who watched the resampling. Handing over a table and
leaving the phrasing to whoever writes the Results section is how "no
detectable difference" turns into "no difference" somewhere between a
spreadsheet and a submitted paper. The two statements are not
interchangeable: the first is what a wide interval supports, the second is a
claim about the world that 49 PDF clusters cannot carry.

So the classification is a function of the corrected p-values, and the
prohibitions travel with the numbers instead of living in someone's memory.

Not a contract-4 deliverable. Plan section 9 picks the paper's title at the
freeze; this is the evidence that choice is made against.

The output is ordered so it can be read top to bottom: what survived, what the
metrics disagree about, then each family in full, then the supporting analyses,
and the prohibitions last because they refer to everything above.
"""

import json
from pathlib import Path

from paper.provenance import git_sha, now_iso

# The level every family is judged at. One number, in one place, so a verdict
# cannot quietly follow a different threshold from the one the paper states.
ALPHA = 0.05

# How each family is introduced. Provenance is part of the spec: two of the four
# metrics were adopted after the primary analysis, and a reader who cannot tell
# which is which cannot judge the multiplicity.
FAMILY_SPECS = (
    ("contrasts", "official weighted macro-F1",
     "The competition's ranking rule. **Pre-specified.**",
     "Whatever else is reported, this is the metric the task is scored on."),
    ("consistent_contrasts", "path-constrained weighted macro-F1 (C-wF1)",
     "The official metric with one change and no others: a field whose "
     "ancestors were not predicted counts as a false prediction instead of "
     "being scored on its own. ⚠️ **Adopted after the primary analysis, not "
     "pre-specified.**",
     "The argument. Because it differs from the official metric in exactly one "
     "respect, a disagreement between the two families localises the effect to "
     "that respect and nothing else."),
    ("hierarchical_contrasts", "hierarchical F1 (hF)",
     "The ancestor-based set metric of the hierarchical-classification "
     "literature. ⚠️ **Adopted after the primary analysis, not pre-specified.**",
     "Answers the reviewer question the study invites: what does an established "
     "structure-aware metric say? ⚠️ It differs from the official metric in "
     "*two* respects — consistency **and** micro versus macro averaging — so it "
     "corroborates the argument but cannot carry it."),
    ("tuple_contrasts", "whole-row tuple accuracy",
     "A row counts only when all four fields are right. **Pre-specified** in "
     "the analysis plan as the secondary reporting metric.",
     "Reported in full because it was planned, including the one contrast that "
     "runs against the methods. Retiring a planned family after seeing which "
     "way it pointed would be selective reporting."),
)


def classify_contrasts(contrasts, alpha=ALPHA) -> dict:
    """Split a pre-specified family by its **Holm-corrected** p-values.

    The five contrasts are tested together, so the verdict has to come from the
    corrected p. The bootstrap percentile interval reported beside each Δ is
    *uncorrected*, and reading significance off it would reintroduce exactly
    the multiplicity Holm is there to remove -- a contrast can miss the
    correction while its raw interval still excludes zero, and in this study
    several do.

    Direction comes from the sign of the point estimate rather than from the
    interval, for the same reason.

    ``undetermined`` is deliberately not called "null": failing to survive the
    correction means the study could not resolve the sign, not that the effect
    is absent.
    """
    better, worse, undetermined = [], [], []
    for key, row in contrasts.items():
        if row.get("p_holm", 1.0) >= alpha:
            undetermined.append(key)
        elif row["delta"] > 0:
            better.append(key)
        else:
            worse.append(key)
    return {"better": better, "worse": worse, "undetermined": undetermined}


def _fmt(row):
    """Δ, its uncorrected interval, and the corrected p the verdict rests on."""
    return (f"{row['delta']:+.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}], "
            f"p_Holm={row.get('p_holm', float('nan')):.3f}"
            f" — {row['description']}")


def _survivors(families) -> list:
    """Every contrast that cleared its own family's correction."""
    rows = []
    for key, title, _, _ in FAMILY_SPECS:
        family = families.get(key)
        if not family:
            continue
        verdict = classify_contrasts(family)
        for contrast in verdict["better"] + verdict["worse"]:
            row = family[contrast]
            rows.append((contrast, title, row))
    return rows


# --------------------------------------------------------------------- header

def _intro(_) -> list:
    return [
        "# Findings brief", "",
        "Generated by `analysis/findings.py` on every `python -m analysis` run. "
        "Every number here comes from the same run that produced `tables/`.", "",
        "**What this file is for.** The tabulars and their captions are the "
        "deliverables; this file says what they license. Where a sentence in "
        "`docs/study_report.md` disagrees with this file, **this file wins** — "
        "it is regenerated, the report is hand-written.", "",
        "**Do not transcribe these figures into the paper.** Take paper numbers "
        "from `tables/*.tex` and `tables/*_caption.txt`, which carry provenance "
        "through `tables/manifest.json`.", "",
    ]


def _how_to_read(_) -> list:
    return [
        "## How to read this file", "",
        "**A contrast is resolved when its Holm-corrected p is below "
        f"{ALPHA}.** Five contrasts are tested together in each family, so a "
        "single uncorrected interval is not the criterion: testing five things "
        "at 5% each gives roughly a 23% chance of at least one false alarm, "
        "which is what the correction removes.", "",
        "**The bracketed intervals are uncorrected 95% bootstrap percentile "
        "intervals.** They describe one contrast in isolation and are printed "
        "because effect size matters, not because they decide anything. "
        "Several contrasts here have an interval excluding zero while failing "
        "the correction.", "",
        "**Three phrasings, three different claims:**", "",
        "| Phrase | Means | Allowed |",
        "|---|---|---|",
        "| no difference | the two are equally good | ❌ never — not shown |",
        "| equivalent | the gap is provably negligible | ❌ never — needs a different test |",
        "| **no detectable difference** | this design could not resolve the sign | ✅ the only one |",
        "",
        "The resampling unit is the source PDF, not the row: 49 clusters, not "
        "2,000 rows. That is the effective sample size behind every interval "
        "below, and the reason so many of them are wide.", "",
    ]


# ------------------------------------------------------------------- sections

def _survived(families) -> list:
    rows = _survivors(families)
    out = ["## 1. What survived the correction", ""]
    if not rows:
        out += ["**Nothing.** No contrast in any family cleared its own Holm "
                f"correction at {ALPHA}.", ""]
        return out
    out += ["| Contrast | Metric | Δ [95% CI, uncorrected] | p_Holm |",
            "|---|---|---|---|"]
    for contrast, title, row in rows:
        out.append(f"| **{contrast}** | {title} | {row['delta']:+.3f} "
                   f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}] | "
                   f"**{row['p_holm']:.3f}** |")
    out.append("")

    official = classify_contrasts(families.get("contrasts", {}))
    if not (official["better"] or official["worse"]) and rows:
        contrasts = sorted({c for c, _, _ in rows})
        out += ["**Read this table as two facts.**", "",
                "1. The metric the competition ranks by resolves **none** of "
                "its five contrasts.",
                f"2. Every metric that accounts for hierarchy consistency "
                f"resolves at least one of the *same* pre-specified contrasts, "
                f"on the *same* resamples, from the *same* predictions "
                f"({', '.join(contrasts)}).", "",
                "Only one of those metrics — C-wF1 — differs from the official "
                "one in a single respect, so it alone licenses attributing the "
                "difference to consistency handling. The other two corroborate "
                "the direction without isolating the cause.", ""]
    return out


def _metric_study(methods) -> list:
    if not methods or not any("hierarchical_mean" in row for row in methods.values()):
        return []
    out = ["## 2. Seven methods under four scoring rules", "",
           "The same predictions, scored four ways. The columns disagree about "
           "which method is best, which is the study's subject rather than an "
           "inconvenience.", "",
           "| Method | official wF1 | C-wF1 | hP | hR | hF | tuple acc. |",
           "|---|---|---|---|---|---|---|"]
    for method, row in methods.items():
        h = row.get("hierarchical_mean")
        if not h:
            continue
        out.append(
            f"| {method} | {row['weighted_macro_f1_mean']:.4f} | "
            f"{row.get('consistent_weighted_macro_f1_mean', float('nan')):.4f} | "
            f"{h['hP']:.4f} | {h['hR']:.4f} | {h['hF']:.4f} | "
            f"{row.get('tuple_exact_match_mean', float('nan')):.4f} |")
    best_official = max(methods, key=lambda m: methods[m]["weighted_macro_f1_mean"])
    worst_official = min(methods, key=lambda m: methods[m]["weighted_macro_f1_mean"])
    best_h = max((m for m in methods if methods[m].get("hierarchical_mean")),
                 key=lambda m: methods[m]["hierarchical_mean"]["hF"])
    out += ["",
            f"**The metrics disagree about the winner: {best_official} on the "
            f"official metric, {best_h} on hF"
            + (f" — and {best_h} is the *worst* method on the official one"
               if best_h == worst_official else "")
            + ".** Always state a ranking together with the metric that "
              "produced it; \"the best method\" is not well defined here.", "",
            "**Reading the columns.** `C-wF1` equals `official wF1` for every "
            "method whose output is legal by construction, and falls below it "
            "only for the unconstrained arm — that equality is the literature's "
            "own check on a path-constrained metric, and it holds here. `hP` "
            "penalises asserting nodes the label does not have, `hR` penalises "
            "omitting nodes it does, and `hF` combines them.", ""]
    return out


def _family_block(number, spec, family) -> list:
    key, title, provenance, role = spec
    verdict = classify_contrasts(family)
    out = [f"### 3.{number} {title}", "", provenance, "", role, ""]
    for contrast in verdict["better"]:
        out.append(f"- **{contrast}** is better: {_fmt(family[contrast])}")
    for contrast in verdict["worse"]:
        out.append(f"- **{contrast}** is *worse*: {_fmt(family[contrast])}")
    for contrast in verdict["undetermined"]:
        out.append(f"- **{contrast}**: no detectable difference — "
                   f"{_fmt(family[contrast])}")
    if not verdict["better"] and not verdict["worse"]:
        out += ["", "**None of the five survived the correction.**"]
    out.append("")
    return out


def _families(families) -> list:
    present = [(spec, families[spec[0]]) for spec in FAMILY_SPECS
               if families.get(spec[0])]
    if not present:
        return []
    out = ["## 3. The Holm families in full", "",
           f"The same five pre-specified contrasts scored under each metric, "
           f"each corrected as its own family of five. **They are not pooled "
           f"into one family of {5 * len(present)}**: the contrasts were "
           "specified once, not once per metric, and pooling would penalise "
           "them for a multiplicity they do not have.", "",
           "The counterpart risk — picking whichever family looks best — is "
           "held off by reporting all five contrasts of every family, "
           "including the ones that run against the methods.", ""]
    for i, (spec, family) in enumerate(present, start=1):
        out += _family_block(i, spec, family)
    return out


def _disagreements(families) -> list:
    primary = families.get("contrasts")
    secondary = families.get("consistent_contrasts")
    if not primary or not secondary:
        return []
    first, second = classify_contrasts(primary), classify_contrasts(secondary)
    crossed = [k for k in primary
               if k in secondary
               and (k in first["undetermined"]) != (k in second["undetermined"])]
    if not crossed:
        return []
    out = ["### Where the two metrics disagree", "",
           "These contrasts are resolved by one family and not the other. The "
           "two metrics differ in exactly one respect — whether a prediction "
           "its own ancestors do not support is scored on its own merits — so "
           "a disagreement localises the effect to that respect and nothing "
           "else. This is the study's central observation, not an "
           "inconsistency to be reconciled.", ""]
    for k in crossed:
        out.append(f"- **{k}**: weighted macro-F1 {_fmt(primary[k])}; "
                   f"path-constrained {_fmt(secondary[k])}")
    out.append("")
    return out


def _secondary_protocol(secondary) -> list:
    """Does the finding hold under the other evaluation target?"""
    if not secondary:
        return []
    families = {k: secondary.get(k) for k, _, _, _ in FAMILY_SPECS}
    rows = _survivors(families)
    out = ["## 4. The same contrasts under the same-document protocol", "",
           "Everything above is the **document-disjoint** protocol, which the "
           "plan designates as primary. The same five contrasts were also run "
           "on the same-document protocol; reporting it matters because a "
           "finding that appears in only one evaluation target is a weaker "
           "finding, and saying so is not optional.", ""]
    if rows:
        out += ["| Contrast | Metric | Δ [95% CI, uncorrected] | p_Holm |",
                "|---|---|---|---|"]
        for contrast, title, row in rows:
            out.append(f"| **{contrast}** | {title} | {row['delta']:+.3f} "
                       f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}] | "
                       f"**{row['p_holm']:.3f}** |")
        out.append("")
    else:
        out += ["**Nothing survived the correction in this protocol.**", ""]

    headline = (families.get("consistent_contrasts") or {}).get("M1-M0")
    if headline:
        resolved = headline.get("p_holm", 1.0) < ALPHA
        out += [
            "⚠️ **The headline contrast does not replicate here.**" if not resolved
            else "**The headline contrast replicates here.**", "",
            f"`M1-M0` on C-wF1 under the same-document protocol: "
            f"{_fmt(headline)}. "
            + ("The direction agrees with the document-disjoint result but the "
               "correction is not cleared, so it must be described as not "
               "replicated rather than replicated weakly. The plausible reason "
               "is that this protocol scores rows from reports the model has "
               "partly seen, where fewer predictions break the hierarchy in the "
               "first place — but that is a hypothesis, not a measurement."
               if not resolved else ""), ""]
    return out


def _regimes(regimes) -> list:
    if not regimes:
        return []
    out = ["## 5. The two evaluation targets (Table 3)", "",
           "Same-document measures seen-report, unseen-paragraph "
           "generalisation and matches the competition's own distribution; "
           "document-disjoint measures generalisation to unseen reports. "
           "**Δ is the gap between two estimation targets, not a bias.**", ""]
    for name, row in regimes.items():
        sig = ("excludes zero" if (row["ci_low"] > 0 or row["ci_high"] < 0)
               else "spans zero")
        out.append(
            f"- **{name}**: same-document {row['same_document']:.3f} vs "
            f"document-disjoint {row['document_disjoint']:.3f}, "
            f"Δ {row['delta']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}] "
            f"({sig})")
    out.append("")
    return out


def _conditional(methods) -> list:
    if not methods or not any("conditional_per_field_mean" in r
                              for r in methods.values()):
        return []
    out = ["## 6. Conditional field F1 (plan §4.5)", "",
           "Each child field scored only on the rows its **gold** parent "
           "admits: `verification_timeline` and `evidence_status` on gold "
           "`promise_status = Yes`, `evidence_quality` on gold "
           "`evidence_status = Yes`. The conditioned rows are identical for "
           "every method, so the columns stay comparable. `promise_status` has "
           "no parent and is unchanged by construction.", "",
           "Reported because the unconditioned score mixes two questions — "
           "choosing the right child label, and repeating the `N/A` the "
           "hierarchy already fixes — and the second is easy for every method. "
           "**The competition still ranks on the unconditioned score**; this "
           "one is diagnostic.", ""]
    fields = list(next(iter(methods.values()))["per_field_mean"])
    out.append("| Method | " + " | ".join(
        f"{f} (all / conditioned)" for f in fields) + " |")
    out.append("|---|" + "---|" * len(fields))
    for method, row in methods.items():
        cells = " | ".join(
            f"{row['per_field_mean'][f]:.3f} / "
            f"{row['conditional_per_field_mean'][f]:.3f}" for f in fields)
        out.append(f"| {method} | {cells} |")
    out.append("")
    return out


def _sensitivity(methods) -> list:
    if not methods or not any("weighted_macro_f1_mean_no_misleading" in r
                              for r in methods.values()):
        return []
    out = ["## 7. `Misleading`: the sensitivity it licenses", "",
           "The official metric recomputed without the two gold `Misleading` "
           "paragraphs. It is the only aggregate the plan permits around a "
           "class with n=2, and it exists to show that no conclusion rests on "
           "those two rows.", "",
           "| Method | weighted macro-F1 | without `Misleading` | Δ |",
           "|---|---|---|---|"]
    for method, row in methods.items():
        free = row.get("weighted_macro_f1_mean_no_misleading")
        if free is None:
            continue
        full = row["weighted_macro_f1_mean"]
        out.append(f"| {method} | {full:.4f} | {free:.4f} | {free - full:+.4f} |")
    out += ["",
            "⚠️ The gap is a property of removing an unlearnable class from a "
            "macro average, **not** a result about the class. The methods move "
            "by nearly the same amount, which is the point: no comparison "
            "between them depends on those two rows.", ""]
    return out


def _instances(cases) -> list:
    runs = (cases or {}).get("runs") or []
    instances = runs[0].get("misleading_cases") if runs else None
    if not instances:
        return []
    out = ["### The two `Misleading` paragraphs, one by one", "",
           "Plan §4.5 allows the per-instance record and nothing aggregated "
           "over two rows.", ""]
    for case in instances:
        guessed = ", ".join(f"{m}→`{p}`" for m, p in case["predicted"].items())
        out.append(f"- `{case['id']}` in {case['pdf_url']}: {guessed}")
    out.append("")
    return out


def _mechanism(cases) -> list:
    if not cases:
        return []
    t = cases["totals"]
    top = max(t["by_rule"].items(), key=lambda kv: kv[1])
    share = (f" ({top[1] / t['n_invalid'] * 100:.0f}%)" if t["n_invalid"] else "")
    out = ["## 8. Failure modes and mechanism (Discussion material)", "",
           f"- Independent argmax emits {t['n_invalid']:,} illegal tuples "
           f"across {t['n_rows']:,} rows ({t['invalid_rate']*100:.2f}%); the "
           f"most common single violation is `{top[0]}` at {top[1]:,}{share}.",
           f"- Projection repairs {t['fields_repaired']:,} fields and destroys "
           f"{t['fields_destroyed']:,}, a net of {t['net_fields']:+,} over "
           f"{t['n_rows']*4:,} field slots.", ""]
    if "na" in t:
        na, sub = t["na"], t["substantive"]
        worst = (min(t["by_class"].items(), key=lambda kv: kv[1]["net"])[0]
                 if t.get("by_class") else None)
        out += [f"- **The exchange is not symmetric.** Repairs land on the "
                f"`N/A` classes ({na['net']:+,}) while the damage falls on the "
                f"substantive ones ({sub['net']:+,})"
                + (f", worst on `{worst}`" if worst else "") + ". macro-F1 "
                "weights every class equally and `N/A` is the easiest class to "
                "predict, so the two nearly cancel in the official metric while "
                "whole-row correctness rises.", ""]
    if "parent_overrides" in t:
        o = t["parent_overrides"]
        out += [f"- **The decoder makes the parent field *more* accurate and "
                f"still loses whole rows.** Searching all 17 states revises "
                f"`promise_status` on {o['n_changed']:,} of {o['n_rows']:,} rows "
                f"relative to the projection on the same probabilities. On the "
                f"field itself that is a net gain: {o['to_correct']:,} become "
                f"correct against {o['to_wrong']:,} broken. On those same rows "
                f"whole-tuple correctness falls from "
                f"{o['tuple_correct_before']:,} to {o['tuple_correct_after']:,}. "
                "The exchange is asymmetric — repairing the parent rarely "
                "rescues the row, because the children are usually still wrong, "
                "while breaking the parent destroys the row outright. This is "
                "the mechanism behind M4-M1, and it is the opposite of the "
                "intuitive story that an unreliable field overturns a reliable "
                "one.", ""]
    runs = cases.get("runs") or []
    if runs and runs[0].get("unobserved"):
        first = runs[0]["unobserved"]
        n_unobs = next(iter(first.values()))["n_unobserved_in_gold"]
        per_method = {m: sum(r["unobserved"][m]["n_emitted_unobserved"]
                             for r in runs) for m in first}
        reached = [m for m, n in per_method.items() if n]
        tally = ", ".join(f"{m} in {n}/{len(runs)} runs"
                          for m, n in per_method.items())
        out += [f"- {n_unobs} legal states never occur in gold. A decoder "
                f"searching the full legal space can reach them: {tally}. "
                + (f"Only the calibrated arms ({', '.join(reached)}) get there, "
                   "which makes this a fact about calibration raising the rare "
                   "classes rather than about the decoder itself."
                   if reached else
                   "None of them does, so the search space is effectively the "
                   "observed states."), ""]
    out += _instances(cases)
    return out


def _unresolved(contrasts) -> list:
    verdict = classify_contrasts(contrasts)
    if not verdict["undetermined"]:
        return []
    return ["## 9. What the official metric could not resolve", "",
            "Not statements that the effect is absent, and several are resolved "
            "by the structure-aware families above. Each contrast below failed "
            "the Holm correction *on the metric the competition ranks by*:", ""] \
        + [f"- **{k}**: no detectable difference — {_fmt(contrasts[k])}"
           for k in verdict["undetermined"]] \
        + ["", "Write these as *no detectable difference*, never as *no "
           "difference* and never as *equivalent*.", ""]


def _prohibitions(audit) -> list:
    dev = audit["development"]
    absent = audit["splits"]["calibration_without_misleading"]
    n_mis = len(audit["splits"]["misleading_rows"])
    return [
        "## 10. Prohibitions", "",
        "Each of these has a specific way of going wrong in a written sentence.", "",
        f"- **`Misleading` (n={n_mis}) must not carry any significance or "
        f"improvement claim.** It is absent from the Calibration partition of "
        f"{absent['n_without']} of the {absent['n_rotations']} rotations, so "
        "most rotations cannot estimate a bias for it at all.",
        "- **Never call a contrast significant because its interval excludes "
        "zero.** The intervals are uncorrected and five contrasts are tested "
        "together; the verdict is `p_Holm`. Several contrasts here have an "
        "interval excluding zero and fail the correction.",
        "- **Never write that the official metric *systematically* "
        "underestimates structured decoding.** The evidence is one benchmark, "
        "one backbone and seven decision rules: it supports *can substantially "
        "understate* or *may fail to reflect improvements in structured-output "
        "validity* on this task, and nothing about the metric in general.",
        "- **The path-constrained and hierarchical families were not "
        "pre-specified.** They were adopted after the primary analysis. Say so "
        "wherever their numbers appear; presenting them as planned analyses "
        "would misrepresent how they were chosen.",
        "- **`±` in Table 2 is seed spread, not a confidence interval.** It "
        "describes the whole pipeline — fold assignment and training together "
        "— and must not be described as model stability.",
        f"- **Δ in Table 3 is the gap between two estimation targets, not a "
        f"bias.** Development and test draw on the same {dev['pdfs']} reports, "
        "so the same-document column is the competition's own distribution, not "
        "a mistake.",
        "- **hF is not a single-factor comparison.** It changes both "
        "consistency handling and micro-versus-macro averaging relative to the "
        "official metric. Only C-wF1 isolates consistency.", "",
    ]


def build_findings(audit, contrasts, regimes, cases=None,
                   consistent_contrasts=None, tuple_contrasts=None,
                   methods=None, hierarchical_contrasts=None,
                   secondary=None) -> str:
    """Assemble the brief. ``secondary`` is the other protocol's summary."""
    families = {
        "contrasts": contrasts,
        "consistent_contrasts": consistent_contrasts,
        "tuple_contrasts": tuple_contrasts,
        "hierarchical_contrasts": hierarchical_contrasts,
    }
    out = []
    out += _intro(audit)
    out += _how_to_read(audit)
    out += _survived(families)
    out += _metric_study(methods)
    out += _families(families)
    out += _disagreements(families)
    out += _secondary_protocol(secondary)
    out += _regimes(regimes)
    out += _conditional(methods)
    out += _sensitivity(methods)
    out += _mechanism(cases)
    out += _unresolved(contrasts)
    out += _prohibitions(audit)
    return "\n".join(out)


def write_findings(out_dir, audit, contrasts, regimes, cases=None,
                   consistent_contrasts=None, tuple_contrasts=None,
                   methods=None, hierarchical_contrasts=None,
                   secondary=None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "findings.md"
    header = f"<!-- generated {now_iso()} from {git_sha()} -->\n\n"
    path.write_text(
        header + build_findings(audit, contrasts, regimes, cases,
                                consistent_contrasts, tuple_contrasts, methods,
                                hierarchical_contrasts, secondary),
        encoding="utf-8")
    return path
