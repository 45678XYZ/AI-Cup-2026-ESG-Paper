r"""Emit contract-4 tables: tabular only, captions beside them, manifest on top.

Contract section 5 splits the responsibility deliberately: content is C's
(number correctness), layout is D's (the 8-page budget). So nothing here emits
``\begin{table}``, ``\caption`` or ``\label``, and only booktabs rules are
used. Column counts follow ``contracts/examples/tables/`` because that is what
D measured the page budget against.

Every cell is computed. The manifest records the sha256 of every input file so
any number in the paper can be traced back to the artifacts that produced it
(contract section 5). "Every input file" means the ones a table's numbers were
actually computed from, which differs per table: table 1 is a dataset audit,
tables 2 and 3 are scored from per-row predictions. Checksumming anything else
yields a manifest that stays identical while an edited input moves every score.
"""

import json
from pathlib import Path

from analysis.audit import SPLITS_DIR
from analysis.bootstrap import N_BOOT
from analysis.findings import ALPHA
from analysis.load import METHODS, predictions_path
from paper.data import REPO_ROOT, TEST_PATH, TRAIN_PATH, VAL_PATH, file_sha256
from paper.labels import EVAL_FIELDS, FIELD_WEIGHTS, FIELDS
from paper.provenance import git_sha, now_iso
from paper.train_config import PROTOCOLS, SEEDS

CONTRACT_VERSION = "1.0"

# Five tabulars, down from seven. ``table5_metrics`` reported the same seven
# rules under six metric columns; once C-wF1 and hF were demoted to prose it
# carried one fact -- that the metrics disagree about the winner -- which is a
# sentence in table 2's caption. ``table7_resolution`` re-presented table 4's
# official block sorted by effect size with the derivation done; that is also a
# sentence, now in table 4's caption. A table that a caption can replace is a
# table the reader has to hold in their head for nothing.
TABLE_FILES = ("table1_dataset.tex", "table2_main.tex",
               "table4_contrasts.tex", "table5_headroom.tex",
               "table6_regimes.tex")

# Tables written by their own analysis module rather than by ``write_tables``,
# because their inputs are not the cross-seed summaries this file consumes.
# Their provenance lives in their own JSON: ``legality_cost.json`` records a
# sha256 per prediction file per arm, which is finer than the manifest's
# per-table list. Listed here so anything that enumerates delivered tables --
# the preview, and D counting floats -- sees all of them from one place.
#
# Keyed by the file each writer produces, and ``EXTERNAL_TABLE_FILES`` is
# derived from it, so a table cannot be listed as delivered without naming
# something that rebuilds it. That gap is not hypothetical: table 7 was
# registered in the list alone and ``python -m analysis`` did not regenerate
# it, while ``analysis/preview.py`` skips a tabular that is missing from disk
# instead of failing -- so the next full rebuild would have dropped a paper
# float with every command still reporting success.
#
# ``kwargs`` names the entry point's arguments each writer wants; both take
# ``out_dir`` positionally.
EXTERNAL_TABLES = {
    "table3_legality_cost.tex": {
        "writer": "analysis.legality_cost:write_legality_cost",
        "kwargs": ("root", "n_boot"),
        "skip_if_absent": True,
    },
    "table7_multilingual_mechanism.tex": {
        "writer": "analysis.multilingual_mechanism:write_report",
        "kwargs": (),
        "skip_if_absent": True,
    },
}

EXTERNAL_TABLE_FILES = tuple(EXTERNAL_TABLES)

# Sorted by number so anything that enumerates the deliverables -- the
# preview, and D reading the directory -- sees them in the order the paper
# presents them. Without the sort the external table lands last whatever its
# number says, which is how a table ends up numbered 3 and printed sixth.
ALL_TABLE_FILES = tuple(sorted(TABLE_FILES + EXTERNAL_TABLE_FILES))

# How each Holm family is named in table 4. Held here rather than imported
# from the brief so the tabular's wording cannot drift with prose edits.
# The two families ``paper_plan.md`` named before any result existed: the
# competition's own metric (section 4.4) and tuple exact-match (section 4.5).
# Path-constrained wF1 and hF were adopted after the primary analysis returned
# its null and are exploratory; ``docs/governance/inference_families.md`` demotes them to
# prose rather than giving each a Holm family, which removes ten tests from the
# page and costs nothing -- what they show, tuple exact-match already shows
# with a pre-specified metric and a ten-times-larger effect.
FAMILY_LABELS = (
    ("contrasts", "Weighted macro-F1 (official)"),
    ("tuple_contrasts", "Tuple accuracy"),
)

# The script that computed the numbers, not the one that formatted them.
SOURCE_SCRIPTS = {
    "table1_dataset.tex": "analysis/audit.py",
    "table2_main.tex": "analysis/aggregate.py",
    "table6_regimes.tex": "analysis/aggregate.py",
    "table4_contrasts.tex": "analysis/aggregate.py",
    "table5_headroom.tex": "analysis/aggregate.py",
}

# The competition test split ships no labels, so its label-derived cells cannot
# be filled. Printing a development number there would be a fabricated cell.
NA = "n/a"

TABLE2_ROWS = (
    ("M0", "None", "Independent"), ("M1", "None", "Projection"),
    ("M2", "Global", "Projection"), ("M3", "Conditional", "Projection"),
    ("M4", "None", "17-state"), ("M5", "Global", "17-state"),
    ("M6", "Conditional", "17-state"),
)

_NUMBER_WORDS = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
                 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def _word(n) -> str:
    """Small counts spelled out, as academic prose wants them."""
    return _NUMBER_WORDS.get(n, f"{n:,}")


def _times(n) -> str:
    return {1: "once", 2: "twice"}.get(n, f"{_word(n)} times")


def _repo_relative(path) -> str:
    """Path as the manifest should record it: relative to the repository.

    ``relative_to`` rather than ``os.path.relpath`` so that a path outside the
    repository fails the containment test instead of being written as a chain
    of ``..`` segments, which would be neither portable nor meaningful. Such a
    path is kept absolute, where it is at least obviously wrong rather than
    quietly relative to nothing.
    """
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def table_inputs(predictions_root, protocols=PROTOCOLS, seeds=SEEDS,
                 methods=METHODS) -> dict:
    """The files each table's numbers actually come from.

    Table 1 is a dataset audit, so it reads the dataset and the split
    manifests. Tables 2 and 3 are scored from per-row predictions -- nothing in
    ``analysis/`` reads ``results/*.json`` at all, so recording those would
    describe an audit trail that does not exist: an edited prediction file
    moves every score while the recorded checksums stay identical.
    """
    audited = [TRAIN_PATH, VAL_PATH, TEST_PATH, *sorted(SPLITS_DIR.glob("*.json"))]
    scored = {
        protocol: [predictions_path(protocol, seed, method, predictions_root)
                   for seed in seeds for method in methods]
        for protocol in protocols
    }
    return {
        # The main table reports the document-disjoint protocol; the
        # same-document protocol reaches the paper through table 3.
        "table1_dataset.tex": audited,
        "table2_main.tex": scored["pdf_group"],
        "table6_regimes.tex": [p for protocol in protocols for p in scored[protocol]],
        "table4_contrasts.tex": scored["pdf_group"],
        "table5_headroom.tex": scored["pdf_group"],
    }


def _survival_summary(families) -> str:
    """How many of each metric's five contrasts survived, spelled out.

    Table 4 prints every contrast; this is the sentence that tells a reader
    what the table adds up to before they read a single row. Counting is over
    the corrected p, never over the intervals -- five contrasts are tested
    together, and an interval that excludes zero decides nothing on its own.
    """
    parts = []
    for key, label in FAMILY_LABELS:
        family = families.get(key)
        if not family:
            continue
        n = sum(1 for row in family.values() if row["p_holm"] < ALPHA)
        parts.append(f"{_word(n)} on {label}")
    if not parts:
        return ""
    listed = ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1]
    return (f" Of the {_word(len(next(iter(families.values()))))} contrasts, "
            f"{listed} survive the correction at $\\alpha$={ALPHA}.")


def _consistency_clause(methods) -> str:
    """Where the secondary metric sits relative to the column the table prints.

    The path-constrained variant has no column of its own, so the caption has
    to locate it. Both halves are computed: how far the unconstrained arm falls
    under it, and whether the constrained arms move at all. The second is the
    literature's own check on such a metric -- a method that guarantees label
    consistency must score identically under both -- and stating it as a
    measured fact rather than a claim is what makes it evidence.
    """
    official = methods["M0"]["weighted_macro_f1_mean"]
    constrained = methods["M0"]["consistent_weighted_macro_f1_mean"]
    others = [m for m in METHODS if m != "M0"]
    worst = max(abs(methods[m]["weighted_macro_f1_mean"]
                    - methods[m]["consistent_weighted_macro_f1_mean"])
                for m in others)
    unchanged = ("M1--M6 score identically under both"
                 if worst < 5e-4 else
                 f"M1--M6 move by at most {worst:.3f}")
    return (f" Under that variant M0 scores {constrained:.3f} against "
            f"{official:.3f} on the official metric, while {unchanged}: only "
            "the unconstrained arm predicts fields its own ancestors do not "
            "support.")


def _ranking_sentence(methods) -> str:
    """Which method each metric prefers -- computed, because the disagreement
    is the point and a stored sentence would survive a rerun that changed it."""
    scored = {m: r for m, r in methods.items() if r.get("hierarchical_mean")}
    if not scored:
        return ""
    best_official = max(scored, key=lambda m: scored[m]["weighted_macro_f1_mean"])
    worst_official = min(scored, key=lambda m: scored[m]["weighted_macro_f1_mean"])
    best_h = max(scored, key=lambda m: scored[m]["hierarchical_mean"]["hF"])
    if best_official == best_h:
        return f" Both metrics rank {best_official} first."
    tail = (f", which is the lowest-scoring rule on the official metric"
            if best_h == worst_official else "")
    return (f" The metrics disagree about the winner: {best_official} on the "
            f"official metric and {best_h} on hF{tail}. A ranking is only "
            "meaningful stated together with the metric that produced it.")


def build_captions(audit, seeds=SEEDS, contrasts=None,
                   consistent_contrasts=None, tuple_contrasts=None,
                   methods=None, hierarchical_contrasts=None) -> dict:
    """Captions computed from the audit rather than stored as text.

    Captions are contract-4 deliverables and reach the paper verbatim, so the
    plan's "所有數字由 script 生成，不手抄" rule covers them exactly as it
    covers the figure's counts. A stored string is correct only until the next
    resample: the audit moves and the caption stays behind, saying something
    that was true last week.
    """
    dev = audit["development"]
    absent = audit["splits"]["calibration_without_misleading"]
    misleading = audit["splits"]["misleading_rows"]
    n_reports = len({row["pdf_url"] for row in misleading})
    return {
        "table1_dataset": (
            "Dataset and split statistics, regenerated from the versioned data "
            f"by analysis/audit.py. Misleading occurs {_times(len(misleading))} in the "
            f"whole development set, in {_word(n_reports)} different reports, and is "
            f"absent from the Calibration partition of {absent['n_without']} of the "
            f"{absent['n_rotations']} rotations. The four fields admit 120 "
            "combinations, of which the hierarchy leaves 17 legal, and no "
            f"development row violates it. All {audit['pdf_overlap']['n_shared']} "
            "development reports also appear in the test split. No company "
            "contributes more than one report, so a document-disjoint split "
            "is also company-disjoint. The "
            "competition test split ships no labels, so its label-derived cells "
            "are marked n/a."
        ),
        "table2_main": (
            "Controlled comparison of decision rules on identical base "
            "probabilities and identical test rows. Each cell is the mean over "
            f"{_word(len(seeds))} seeds of a single score computed on all "
            f"{dev['paragraphs']:,} concatenated test rows; $\\pm$ is the sample "
            "standard deviation across seeds, which reflects the variability of "
            "the whole pipeline -- fold assignment and training together -- "
            "rather than model stability."
            + (" Paired contrasts between these rows, under each metric and "
               "with their Holm-corrected p-values, are reported in Table 4."
               if contrasts else "")
            + (_consistency_clause(methods) + _ranking_sentence(methods)
               if methods else "")
        ),
        "table5_headroom": (
            f"Where the official score falls short under {HEADROOM_METHOD}, and "
            "what closing each part is worth. Shortfall is the field's weight "
            "times its distance from a perfect macro-F1; share is its portion "
            "of the total. The last column is what a 0.1 gain on a single "
            "class's F1 adds to the weighted total -- the field's weight "
            "divided by its class count, times 0.1 -- which orders the fields "
            "differently from the shortfall: macro-F1 averages over classes, so "
            "a field carrying more of them returns less per class improved. "
            "\\textbf{Not all of the shortfall is reachable.} Misleading occurs "
            "twice in the development set and is never predicted by any arm, so "
            "a quarter of evidence\\_quality's macro-F1 -- 0.0875 of the "
            "weighted total -- is out of reach by construction."
        ),
        "table4_contrasts": (
            "The five pre-specified contrasts of the analysis plan, under the "
            "two metrics the plan named before any result existed: the "
            "competition's own weighted macro-F1 and tuple exact-match. Paired "
            f"PDF-cluster bootstrap over {N_BOOT:,} resamples on the "
            "document-disjoint protocol; the resampling unit is the source "
            "report, not the paragraph. Each metric is Holm-corrected as its "
            "own family of five rather than pooled: the contrasts were "
            "specified once, not once per metric. "
            "\\textbf{Bold $p_{\\mathrm{Holm}}$ marks a contrast that survives "
            "the correction; the bracketed intervals are uncorrected 95\\% "
            "percentile intervals and decide nothing on their own.} Two "
            "further structure-aware metrics were examined after the primary "
            "analysis and are reported as exploratory in the text, not as Holm "
            "families here."
            + _resolution_clause(contrasts, methods)
            + _survival_summary({
                "contrasts": contrasts,
                "tuple_contrasts": tuple_contrasts,
            })
            + " The official metric and tuple accuracy were named in the "
            "analysis plan in advance; the path-constrained variant and the "
            "hierarchical F1 were adopted after the primary analysis and are "
            "not pre-specified."
        ),
        "table6_regimes": (
            "Same-document versus document-disjoint evaluation. The left column "
            "measures seen-report, unseen-paragraph generalisation and matches "
            "the competition distribution, since test and development draw on "
            f"the same {dev['pdfs']} reports; the right column measures "
            "generalisation to entirely unseen reports. $\\Delta$ is the gap "
            "between two estimation targets and is not a bias estimate. All "
            "seven decision rules are reported rather than a best-scoring "
            "subset: every $\\Delta$ is positive and every interval excludes "
            "zero, so the gap is a property of the split and not of any one "
            "rule."
        ),
    }


def _resolution_clause(contrasts, methods) -> str:
    """What magnitude this benchmark can actually resolve, in one sentence.

    This used to be a table of its own, which re-presented the official block
    above sorted by effect size with the arithmetic done. Everything it showed
    is already in the rows: the sentence names the one contrast whose interval
    clears zero without surviving, and puts it beside the full spread across
    the rules the study set out to separate.

    Two things it has to be careful about. It scans the official family only,
    while the table beside it also prints tuple accuracy -- where M1-M0 is
    +0.035, four times what this sentence calls the largest -- so the metric is
    named. And the spread it offers as scale is sometimes the very contrast it
    is scaling: when the two rules named are the best- and worst-scoring of the
    set, ``_method_span`` returns this contrast's own magnitude, and printing
    it after "for scale" reads as a second, corroborating measurement.
    """
    if not contrasts:
        return ""
    shaped = [(abs(r["delta"]), k, r) for k, r in contrasts.items()
              if (r["ci_low"] > 0 or r["ci_high"] < 0) and r["p_holm"] >= ALPHA]
    if not shaped:
        return ""
    magnitude, key, row = max(shaped)
    head = (f" The largest effect on the official metric, {key} at "
            f"$|\\Delta|$ = {magnitude:.3f}, has an interval excluding zero "
            f"and still does not survive the correction "
            f"($p_{{\\mathrm{{Holm}}}}$ = {row['p_holm']:.3f})")
    if not methods:
        return head + "."
    if _spans_the_extremes(key, methods):
        return (f"{head} -- and those two rules are the highest- and "
                f"lowest-scoring of the {_word(len(methods))}, so that is not "
                f"one contrast among many but the entire spread the benchmark "
                f"has to resolve.")
    return (f"{head}; for scale, the full range across the "
            f"{_word(len(methods))} decision rules is "
            f"{_method_span(methods):.4f}.")


def _spans_the_extremes(key, methods) -> bool:
    """Whether the contrast's two rules are the best- and worst-scoring ones.

    When they are, ``_method_span`` is |delta| of this same contrast and the
    two numbers are one measurement, not two.
    """
    named = set(key.split("-"))
    if len(named) != 2 or not named.issubset(methods):
        return False
    scores = {m: v["weighted_macro_f1_mean"] for m, v in methods.items()}
    return named == {max(scores, key=scores.get), min(scores, key=scores.get)}


def _method_span(methods) -> float:
    """Spread of the official metric across every decision rule.

    The resolution clause needs something to be read against: a minimum
    detectable difference is a number until it is put beside the difference
    the study was built to detect. Only meaningful as scale when the contrast
    being scaled is not itself the span -- see ``_spans_the_extremes``.
    """
    if not methods:
        return 0.0
    scores = [m["weighted_macro_f1_mean"] for m in methods.values()]
    return max(scores) - min(scores)


def _f(value, places=3):
    return f"{value:.{places}f}"


def _pm(mean, std):
    return f"{_f(mean)}$\\pm${_f(std)}"


def _tabular(spec, header, body_lines):
    rows = "\n".join(body_lines)
    return (
        f"\\begin{{tabular}}{{{spec}}}\n"
        "\\toprule\n"
        f"{header} \\\\\n"
        "\\midrule\n"
        f"{rows}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


def render_table1(audit) -> str:
    dev, test = audit["development"], audit["test"]

    def row(label, dev_value, test_value):
        return f"{label} & {dev_value} & {test_value} \\\\"

    def yes_no(value):
        return "Yes" if value else "No"

    def valid_gold(partition):
        if not partition["labelled"]:
            return NA
        valid = partition["paragraphs"] - partition["invalid_rows"]
        return f"{valid} / {partition['paragraphs']}"

    eq = dev["class_support"]["evidence_quality"]
    vt = dev["class_support"]["verification_timeline"]
    # The overlap is the reason Table 3 exists: development and test are drawn
    # from the same reports, so the competition's own split measures
    # seen-report generalisation. Printing it as a row rather than leaving it
    # to prose is what C's remit asks for.
    shared = audit["pdf_overlap"]["n_shared"]
    body = [
        row("Paragraphs", dev["paragraphs"], test["paragraphs"]),
        row("Source reports (PDFs)", dev["pdfs"], test["pdfs"]),
        # Stays in the tabular by remit, not by taste: plan section 7 requires
        # the 100% dev/test overlap to be prominent rather than left to a
        # caption, because it is the reason Table 3 exists at all.
        row("\\quad shared with other split", shared, shared),
        row("Companies", dev["companies"], test["companies"]),
        row("Gold labels available", yes_no(dev["labelled"]),
            yes_no(test["labelled"])),
        row("Gold hierarchy-valid tuples", valid_gold(dev), valid_gold(test)),
        row("Legal states observed", f"{dev['legal_states_observed']} / 17", NA),
        "\\midrule",
        "\\multicolumn{3}{l}{\\emph{Rarest classes}} \\\\",
        row("\\quad within\\_2\\_years", vt["within_2_years"], NA),
        row("\\quad Misleading", eq["Misleading"], NA),
    ]
    return _tabular(
        "lrr", "Statistic & Development & Competition test", body,
    )


def render_table2(summary) -> str:
    """Seven decision rules over one set of probabilities.

    The four per-field columns are descriptive -- no claim rests on them, and
    table 8 gives that breakdown once with the weights and marginal values that
    make it mean something. They stay because the shape is declared in
    ``contracts/examples/tables/table2_main.tex``: D builds against it, and a
    column count is exactly the kind of thing that is cheap to change here and
    expensive to discover downstream. Dropping them is a contract change, not
    a rendering choice.
    """
    body = []
    for method, calibration, decoding in TABLE2_ROWS:
        row = summary["methods"][method]
        fields = " & ".join(_f(row["per_field_mean"][f]) for f in FIELDS)
        body.append(
            f"{method} & {calibration} & {decoding} & "
            f"{_pm(row['weighted_macro_f1_mean'], row['weighted_macro_f1_std'])} & "
            f"{fields} & {_f(row['tuple_exact_match_mean'])} & "
            f"{_f(row['invalid_tuple_rate_mean'] * 100, 1)} \\\\"
        )
    # ``wF1 (official)`` rather than ``Weighted F1``: the path-constrained
    # variant in table 4 is also a weighted F1, so the bare short form does not
    # identify the metric the competition ranks by. ``Tuple acc.`` is the same
    # quantity tables 3 and 7 print, spelled the same way -- these are the two
    # columns a reader carries between them.
    header = ("ID & Calibration & Decoding & wF1 (official) & PS & VT & ES & EQ "
              "& Tuple acc. & Invalid \\%")
    return _tabular("llrrrrrrrr", header, body)


def render_table4(summary) -> str:
    """Every pre-specified contrast under every metric, with its corrected p.

    This is the study's central result and it used to live in table 2's
    caption, which had grown past 500 words and still had no room for the
    fourth family. A four-by-five statistical result is not caption material:
    a reader has to be able to scan down the official metric's column, see no
    bold, then scan the others and see it.
    """
    body = []
    for key, label in FAMILY_LABELS:
        family = summary.get(key)
        if not family:
            continue
        if body:
            body.append("\\midrule")
        for i, (contrast, row) in enumerate(family.items()):
            p_holm = f"{row['p_holm']:.3f}"
            if row["p_holm"] < ALPHA:
                p_holm = f"\\textbf{{{p_holm}}}"
            body.append(
                f"{label if i == 0 else ''} & {contrast} & {row['delta']:+.3f} & "
                f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}] & {p_holm} \\\\")
    header = ("Metric & Contrast & $\\Delta$ & 95\\% CI & "
              "$p_{\\mathrm{Holm}}$")
    return _tabular("llrrr", header, body)



def render_table6_regimes(regimes) -> str:
    body = []
    for label, row in regimes.items():
        ci = f"[{_f(row['ci_low'])}, {_f(row['ci_high'])}]"
        body.append(
            f"{label} & {_f(row['same_document'])} & "
            f"{_f(row['document_disjoint'])} & {_f(row['delta'])} & {ci} \\\\"
        )
    header = "Method & Same-document & Document-disjoint & $\\Delta$ & 95\\% CI"
    return _tabular("lrrrr", header, body)



HEADROOM_METHOD = "M1"


def render_table5_headroom(summary) -> str:
    """Where the official score's shortfall sits, and what closing it is worth.

    Two columns that rank the fields differently. Shortfall is what a
    participant would read to decide where to work; marginal value is what
    that work would actually return, and macro-F1 divides by the class count,
    so a field carrying more classes returns less per class improved. Ordering
    by shortfall and printing the marginal value beside it is the whole table:
    the column a reader would sort on is not the column that decides.

    The last column is ``weight / |classes| * 0.1`` -- a gain of 0.1 on *one*
    class's F1, not on the field's macro-F1, which would be ``weight * 0.1``
    and would carry no class count at all. Naming it for the field would make
    the column four times larger for evidence_quality and, worse, would delete
    the reason the two orderings differ.

    Reported for M1, the rule this study recommends. Naming it rather than
    taking the best-scoring arm keeps a score out of the choice.
    """
    per_field = summary["methods"][HEADROOM_METHOD]["per_field_mean"]
    total = sum(FIELD_WEIGHTS[f] * (1 - per_field[f]) for f in FIELDS)

    rows = []
    for field in FIELDS:
        weight = FIELD_WEIGHTS[field]
        shortfall = weight * (1 - per_field[field])
        rows.append((field, weight, per_field[field], shortfall,
                     shortfall / total if total else 0.0,
                     weight / len(EVAL_FIELDS[field]) * 0.10))
    rows.sort(key=lambda r: -r[3])

    body = [
        f"{f.replace('_', chr(92) + '_')} & {w:.2f} & {mf:.3f} & {sf:.3f} & "
        f"{share * 100:.1f}\\% & {mv:.4f} \\\\"
        for f, w, mf, sf, share, mv in rows
    ]
    header = ("Field & Weight & macro-F1 & Shortfall & Share & "
              "Value of $+0.1$ per class")
    return _tabular("lrrrrr", header, body)


def write_tables(out_dir, audit, summaries, regimes, inputs_by_table,
                 seeds=SEEDS) -> Path:
    """Write the three tabulars, their captions and the provenance manifest.

    ``inputs_by_table`` maps each table file to the inputs its numbers were
    computed from -- see ``table_inputs``. An empty list is refused rather than
    written: the manifest is the claim-evidence audit trail, and an empty one
    asserts nothing while looking exactly like a complete one.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in TABLE_FILES:
        if not inputs_by_table.get(name):
            raise ValueError(
                f"no input files recorded for {name}. The manifest is the "
                "claim-evidence audit trail (contract §5); writing an empty "
                "one would let every number in the paper trace back to nothing."
            )

    rendered = {
        "table1_dataset.tex": render_table1(audit),
        # The main table reports the document-disjoint protocol; the
        # same-document protocol reaches the paper through Table 3.
        "table2_main.tex": render_table2(summaries["pdf_group"]),
        "table6_regimes.tex": render_table6_regimes(regimes),
        "table4_contrasts.tex": render_table4(summaries["pdf_group"]),
        "table5_headroom.tex": render_table5_headroom(summaries["pdf_group"]),
    }
    for name, content in rendered.items():
        (out_dir / name).write_text(content, encoding="utf-8")
    # Table 2 reports the document-disjoint protocol, so its contrasts are
    # the ones that belong under it.
    contrasts = summaries["pdf_group"]["contrasts"]
    captions = build_captions(
        audit, seeds=seeds, contrasts=contrasts,
        consistent_contrasts=summaries["pdf_group"].get("consistent_contrasts"),
        tuple_contrasts=summaries["pdf_group"].get("tuple_contrasts"),
        hierarchical_contrasts=summaries["pdf_group"].get("hierarchical_contrasts"),
        methods=summaries["pdf_group"]["methods"])
    for stem, caption in captions.items():
        (out_dir / f"{stem}_caption.txt").write_text(caption + "\n", encoding="utf-8")

    def entry(name):
        paths = [Path(p) for p in inputs_by_table[name]]
        # Record repo-relative, hash the path as given: the manifest is
        # committed and read from a fresh clone, where this machine's absolute
        # paths mean nothing, but the checksum has to open the actual file.
        recorded = [_repo_relative(p) for p in paths]
        return {
            "source_script": SOURCE_SCRIPTS[name],
            "input_files": recorded,
            "input_sha256": {r: file_sha256(p) for r, p in zip(recorded, paths)},
        }

    manifest = {
        "contract_version": CONTRACT_VERSION,
        "generated_at": now_iso(),
        "git_sha": git_sha(),
        "tables": {name: entry(name) for name in TABLE_FILES},
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    return out_dir
