#!/usr/bin/env python3
"""Stage 3 (one molecule): MD output -> viz_data.json + explorer HTML + features.

    conda run -n mdsprep python scripts/20_export_viz.py --md-dir data/md_demo/1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ensemble_qsar.features import aggregate  # noqa: E402
from ensemble_qsar.viz import build_html, export  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--md-dir", required=True, type=Path, help="a Stage-2 output dir")
    ap.add_argument("--equilibration-frame", type=int, default=0)
    args = ap.parse_args()

    print(f"exporting {args.md_dir} ...")
    viz = export.build_viz_data(args.md_dir, equilibration_frame=args.equilibration_frame)
    mol_id = viz["meta"]["mol_id"]

    (args.md_dir / "viz_data.json").write_text(__import__("json").dumps(viz))
    html = build_html.build_html(viz, args.md_dir / f"explore_{mol_id}.html")
    aggregate.write_feature_detail(viz, args.md_dir / "feature_detail.json")

    print(f"  mol_id: {mol_id}  frames: {viz['meta']['n_frames']}  clusters: {len(viz['clusters'])}")
    print(f"  images/cluster: {len(viz['clusters'][0]['images']) if viz['clusters'] else 0}")
    print(f"  -> {html}")
    print(f"  -> {args.md_dir / 'viz_data.json'}")
    print(f"  -> {args.md_dir / 'feature_detail.json'}")


if __name__ == "__main__":
    main()
