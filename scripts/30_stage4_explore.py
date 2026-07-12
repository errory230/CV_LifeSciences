#!/usr/bin/env python3
"""Stage 4 (exploratory): ensemble feature dispersion vs. flexibility.

Auto-detects completed molecules under --md-root (those with viz_data.json or
analysis/metrics.csv), computes each molecule's per-frame descriptor dispersion,
joins Stage-1 flexibility + Caco-2 label, and writes Analysis-1 (scatter) + the
summary table. No model is trained.

    python scripts/30_stage4_explore.py \
        --md-root data/md_caco2 \
        --stage1 data/processed/caco2_wang/representative_set.csv \
        --outdir results/stage4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ensemble_qsar.analysis import stage4  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--md-root", required=True, type=Path)
    ap.add_argument("--stage1", type=Path,
                    default=ROOT / "data/processed/caco2_wang/representative_set.csv")
    ap.add_argument("--outdir", type=Path, default=ROOT / "results/stage4")
    ap.add_argument("--burnin-frac", type=float, default=0.0,
                    help="discard this fraction of leading frames as burn-in")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    table, pending = stage4.detect_and_tabulate(
        args.md_root, args.stage1, burnin_frac=args.burnin_frac)
    print(f"completed molecules: {len(table)} | pending: {len(pending)}")
    if pending:
        print("  pending:", ", ".join(pending[:20]) + (" …" if len(pending) > 20 else ""))
    if table.empty:
        print("no completed molecules with extracted features yet."); return

    table.to_csv(args.outdir / "ensemble_dispersion_table.csv", index=False)
    png = stage4.plot_analysis1(table, args.outdir / "analysis1_dispersion_vs_flexibility.png")
    print(f"\nAnalysis 1 -> {png}")
    print(f"table      -> {args.outdir / 'ensemble_dispersion_table.csv'}")
    # quick text preview by flex class
    if "flex_class" in table:
        g = table.groupby("flex_class", observed=True)["psa3d_std"].agg(["count", "mean"])
        print("\nmean per-frame 3D-PSA std by flex_class:")
        print(g.to_string())


if __name__ == "__main__":
    main()
