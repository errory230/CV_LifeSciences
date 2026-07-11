#!/usr/bin/env python3
"""Stage 3 (all molecules): loop Stage-2 outputs, build explorers + features.csv.

Processes every molecule directory under --md-root that has a completed analysis
(analysis/metrics.csv), fail-soft, then writes an index.html and a combined
features.csv. Molecules still running in Stage 2 are simply skipped.

    conda run -n mdsprep python scripts/21_build_all.py --md-root data/md_demo
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ensemble_qsar.features import aggregate  # noqa: E402
from ensemble_qsar.viz import build_html, export  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--md-root", required=True, type=Path)
    ap.add_argument("--equilibration-frame", type=int, default=0)
    args = ap.parse_args()

    dirs = sorted(d for d in args.md_root.iterdir()
                  if (d / "analysis" / "metrics.csv").exists())
    print(f"found {len(dirs)} completed molecule(s)")

    viz_list, entries = [], []
    for d in dirs:
        try:
            viz = export.build_viz_data(d, equilibration_frame=args.equilibration_frame)
            mol_id = viz["meta"]["mol_id"]
            (d / "viz_data.json").write_text(json.dumps(viz))
            href = str((d / f"explore_{mol_id}.html").resolve())
            build_html.build_html(viz, d / f"explore_{mol_id}.html")
            aggregate.write_feature_detail(viz, d / "feature_detail.json")
            viz_list.append(viz)
            entries.append({"mol_id": mol_id, "href": href,
                            "n_frames": viz["meta"]["n_frames"], "n_clusters": len(viz["clusters"])})
            print(f"  [ok] {mol_id}")
        except Exception as e:  # noqa: BLE001 fail-soft
            print(f"  [FAILED] {d.name}: {type(e).__name__}: {e}")
            (args.md_root / f"_error_{d.name}.txt").write_text(traceback.format_exc())

    if viz_list:
        build_html.build_index(entries, args.md_root / "index.html")
        aggregate.build_features_table(viz_list).to_csv(args.md_root / "features.csv", index=False)
        print(f"\nwrote {args.md_root/'index.html'} and {args.md_root/'features.csv'} "
              f"({len(viz_list)} molecules)")


if __name__ == "__main__":
    main()
