"""One command to regenerate every number C is responsible for.

    python -m analysis --predictions-root contracts/examples   # synthetic
    python -m analysis                                          # real results

Plan section 6.2 lists a one-command table rebuild as required work before the
main experiment. It matters most after the 8/23 results freeze, when numbers may
only be recomputed and never changed: this command, run against the frozen
artifacts, must reproduce the tables exactly.
"""

import argparse
import json
from importlib import import_module
from pathlib import Path

from analysis.aggregate import protocol_summary, regime_comparison
from analysis.audit import full_audit
from analysis.cases import write_case_analysis
from analysis.findings import write_findings
from analysis.bootstrap import BOOTSTRAP_SEED, N_BOOT
from analysis.figure1 import build as build_figure1, tex_available
from analysis.figure2 import build as build_figure2
from analysis.load import EXAMPLES_ROOT, REAL_ROOT, pdf_clusters
from analysis.tables import EXTERNAL_TABLES, table_inputs, write_tables
from paper.data import REPO_ROOT, canonical_row_order, load_dev
from paper.train_config import PROTOCOLS, SEEDS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--predictions-root", type=Path, default=REAL_ROOT,
                    help="directory holding predictions/; every score comes "
                         "from there. Pass contracts/examples to run against "
                         "the synthetic set")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "tables")
    ap.add_argument("--figures-dir", type=Path, default=REPO_ROOT / "figures")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    ap.add_argument("--summaries-out", type=Path, default=None,
                    help="also dump the full summary objects as JSON")
    args = ap.parse_args()

    dev = load_dev()
    order = canonical_row_order(dev)
    clusters = pdf_clusters(order, dev)

    audit = full_audit(dev)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.out_dir / "audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=1)
    print(f"audit   -> {audit_path}")

    summaries = {}
    for protocol in PROTOCOLS:
        summaries[protocol] = protocol_summary(
            protocol, order, args.predictions_root, clusters,
            n_boot=args.n_boot, seeds=SEEDS, dev=dev,
            bootstrap_seed=args.bootstrap_seed,
        )
        print(f"summary -> {protocol}")

    regimes = regime_comparison(
        summaries, order, args.predictions_root, clusters,
        n_boot=args.n_boot, seeds=SEEDS, bootstrap_seed=args.bootstrap_seed,
    )

    if args.summaries_out is not None:
        args.summaries_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.summaries_out, "w", encoding="utf-8") as f:
            json.dump({"summaries": summaries, "regimes": regimes}, f,
                      ensure_ascii=False, indent=1)
        print(f"stats   -> {args.summaries_out}")

    # Each table records the files its own numbers came from: the dataset and
    # split manifests for table 1, the per-row predictions for tables 2 and 3.
    cases = write_case_analysis(args.out_dir, order, args.predictions_root,
                                dev=dev, seeds=SEEDS)
    print(f"cases   -> {cases}")

    # What the intervals license D to write. Not a contract-4 deliverable;
    # plan §9 picks the title at the freeze and this is its evidence.
    brief = write_findings(args.out_dir, audit,
                           summaries["pdf_group"]["contrasts"], regimes,
                           cases=json.loads(cases.read_text(encoding="utf-8")),
                           consistent_contrasts=summaries["pdf_group"]["consistent_contrasts"],
                           tuple_contrasts=summaries["pdf_group"]["tuple_contrasts"],
                           hierarchical_contrasts=summaries["pdf_group"]["hierarchical_contrasts"],
                           methods=summaries["pdf_group"]["methods"],
                           secondary=summaries["row_strat"])
    print(f"brief   -> {brief}")

    inputs = table_inputs(args.predictions_root, seeds=SEEDS)
    write_tables(args.out_dir, audit, summaries, regimes, inputs, seeds=SEEDS)
    print(f"tables  -> {args.out_dir}")

    # Tables whose inputs are not the cross-seed summaries above write
    # themselves. They are driven from the registry rather than called by name
    # so that registering one is the same act as rebuilding it -- listing a
    # table without wiring its writer is how a paper float silently stops being
    # regenerated. Each is skipped rather than failed when its inputs are
    # absent: a checkout carrying only the frozen anchor -- which is what
    # contract section 4 promises -- can still rebuild every other deliverable.
    available = {"root": args.predictions_root, "n_boot": args.n_boot}
    for name, spec in EXTERNAL_TABLES.items():
        module_name, func_name = spec["writer"].split(":")
        writer = getattr(import_module(module_name), func_name)
        kwargs = {k: available[k] for k in spec["kwargs"] if k in available}
        try:
            writer(args.out_dir, **kwargs)
            print(f"{name:34}-> {args.out_dir}")
        except FileNotFoundError as missing:
            if not spec["skip_if_absent"]:
                raise
            print(f"{name:34}-> skipped, an input is absent from this "
                  f"checkout: {missing}")

    # Both figures are TikZ, so rebuilding them needs TeX. Figure 1's counts
    # come from paper.labels; Figure 2 consumes the external multilingual table
    # written by the registry above.
    if tex_available():
        figure1 = build_figure1(args.figures_dir / "figure1_hierarchy.pdf")
        print(f"figure  -> {figure1}")
        multilingual_table = args.out_dir / "table7_multilingual_mechanism.tex"
        if multilingual_table.is_file():
            figure2 = build_figure2(
                multilingual_table,
                args.figures_dir / "figure2_multilingual.pdf",
            )
            print(f"figure  -> {figure2}")
        else:
            print("figure2 -> skipped, table7_multilingual_mechanism.tex "
                  "was not generated")
    else:
        print("figure  -> skipped, no TeX compiler on PATH; the committed "
              "figure PDFs are unchanged and still current")

    if Path(args.predictions_root).resolve() == EXAMPLES_ROOT.resolve():
        print("\nINPUTS WERE SYNTHETIC. Every score above is fabricated and "
              "must never reach the paper; only the shapes are meaningful.")


if __name__ == "__main__":
    main()
