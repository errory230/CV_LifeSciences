#!/usr/bin/env python3
"""Build the catalog hub (index.html + hub_index.json) over all explorers.

Idempotent: scans md-root, collects every molecule that has a viz_data.json, and
lists the rest as pending. Re-run any time to refresh.

    conda run -n mdsprep python scripts/22_build_hub.py --md-root data/md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ensemble_qsar.viz import build_hub  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--md-root", required=True, type=Path)
    args = ap.parse_args()

    html_path, json_path = build_hub.build_hub(args.md_root)
    hub = json.loads(json_path.read_text())
    print(f"collected {hub['n_molecules']} molecule(s), {hub['n_pending']} pending")
    print(f"  class distribution: {hub['class_distribution']}")
    print(f"  -> {html_path}")
    print(f"  -> {json_path}")


if __name__ == "__main__":
    main()
