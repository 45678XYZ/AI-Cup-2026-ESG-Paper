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
from analysis.bootstrap import BOOTSTRAP_SEED, N_BOOT
from analysis.figure1 import build as build_figure1
from analysis.load import EXAMPLES_ROOT, REAL_ROOT, pdf_clusters
from analysis.tables import write_tables
from paper.data import REPO_ROOT, canonical_row_order, load_dev
from paper.train_config import PROTOCOLS, SEEDS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--predictions-root", type=Path, default=REAL_ROOT,
                    help="directory holding predictions/ and results/; pass "
                         "contracts/examples to run against the synthetic set")
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

    inputs = sorted((Path(args.predictions_root) / "results").glob("*.json"))
    write_tables(args.out_dir, audit, summaries, regimes, inputs)
    print(f"tables  -> {args.out_dir}")

    figure = build_figure1(args.figures_dir / "figure1_hierarchy.pdf")
    print(f"figure  -> {figure}")

    if Path(args.predictions_root).resolve() == EXAMPLES_ROOT.resolve():
        print("\nINPUTS WERE SYNTHETIC. Every score above is fabricated and "
              "must never reach the paper; only the shapes are meaningful.")


if __name__ == "__main__":
    main()
