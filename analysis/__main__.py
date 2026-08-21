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
from pathlib import Path

from analysis.aggregate import protocol_summary, regime_comparison
from analysis.audit import full_audit
from analysis.cases import write_case_analysis
from analysis.findings import write_findings
from analysis.bootstrap import BOOTSTRAP_SEED, N_BOOT
from analysis.figure1 import build as build_figure1, tex_available
from analysis.load import EXAMPLES_ROOT, REAL_ROOT, pdf_clusters
from analysis.tables import table_inputs, write_tables
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
                           cases=json.loads(cases.read_text(encoding="utf-8")))
    print(f"brief   -> {brief}")

    inputs = table_inputs(args.predictions_root, seeds=SEEDS)
    write_tables(args.out_dir, audit, summaries, regimes, inputs, seeds=SEEDS)
    print(f"tables  -> {args.out_dir}")

    # The figure is TikZ, so rebuilding it needs TeX. Its counts come from
    # paper.labels rather than from this run, so skipping it cannot put the
    # committed PDF out of step with the tables printed above.
    if tex_available():
        figure = build_figure1(args.figures_dir / "figure1_hierarchy.pdf")
        print(f"figure  -> {figure}")
    else:
        print("figure  -> skipped, no latexmk on PATH; the committed "
              "figures/figure1_hierarchy.pdf is unchanged and still current")

    if Path(args.predictions_root).resolve() == EXAMPLES_ROOT.resolve():
        print("\nINPUTS WERE SYNTHETIC. Every score above is fabricated and "
              "must never reach the paper; only the shapes are meaningful.")


if __name__ == "__main__":
    main()
