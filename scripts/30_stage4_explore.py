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

import numpy as np

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
    ap.add_argument("--prep-root", type=Path, default=ROOT / "data/prep/caco2",
                    help="Stage-2a prep dirs (for the static single-conformer 3D-PSA)")
    ap.add_argument("--burnin-frac", type=float, default=0.0,
                    help="discard this fraction of leading frames as burn-in")
    ap.add_argument("--case-molecules", nargs="+", default=None,
                    help="explicit mol_ids for Analysis 2 (default: auto rigid+flexible)")
    ap.add_argument("--n-each", type=int, default=2, help="Analysis 2 molecules per extreme")
    ap.add_argument("--show-tpsa", action="store_true",
                    help="also overlay static 2D-TPSA (note: different scale from 3D-PSA)")
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
    png1 = stage4.plot_analysis1(table, args.outdir / "analysis1_dispersion_vs_flexibility.png")
    print(f"\nAnalysis 1 -> {png1}  (3D-PSA + Rg dispersion vs flexibility)")
    print(f"table      -> {args.outdir / 'ensemble_dispersion_table.csv'}")

    # Analysis 2: case-study distributions (auto rigid vs flexible, or explicit)
    cases = stage4.case_study(args.md_root, table, mol_ids=args.case_molecules, n_each=args.n_each)
    if cases:
        png2 = stage4.plot_analysis2(cases, args.outdir / "analysis2_casestudy_distributions.png")
        print(f"Analysis 2 -> {png2}  (cases: {', '.join(c['mol_id'] for c in cases)})")

    # Analysis 2 (all molecules): 3D-PSA distribution stack sorted by Kier φ, with
    # the static single-conformer 3D-PSA overlaid (same units as the ensemble).
    dcases = stage4.distribution_cases(args.md_root, table, prep_root=args.prep_root)
    if dcases:
        png2b = stage4.plot_distribution_stack(
            dcases, args.outdir / "analysis2_distribution_stack.png", show_tpsa=args.show_tpsa)
        n_static = sum(c["static_psa3d"] is not None for c in dcases)
        n_out = sum(c["static_psa3d"] is not None and
                    not (np.percentile(c["psa3d_frames"], 2.5) <= c["static_psa3d"]
                         <= np.percentile(c["psa3d_frames"], 97.5)) for c in dcases)
        print(f"Analysis 2b -> {png2b}  ({len(dcases)} molecules, "
              f"{n_static} with static 3D-PSA, {n_out} static point outside 95% band)")

    # quick text preview by flex class
    for d in ("psa3d", "rg"):
        col = f"{d}_std"
        if col in table:
            g = table.groupby("flex_class", observed=True)[col].agg(["count", "mean"])
            print(f"\nmean per-frame {d} std by flex_class:\n{g.to_string()}")

    # Analysis 3: monotonic-trend statistic (Spearman) + summary/limitations report
    rep = stage4.write_report(table, args.outdir)
    print("\nAnalysis 3: Spearman trend (dispersion std vs continuous flexibility)")
    print(rep["trends"].to_string(index=False))
    print(f"\nAnalysis 3 -> {rep['report']}")
    print(f"           -> {rep['summary_csv']}")
    print(f"           -> {rep['trends_csv']}")


if __name__ == "__main__":
    main()
