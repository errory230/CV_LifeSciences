#!/usr/bin/env python3
"""MDS-Ensemble QSAR — SMILES -> MD-ensemble conformation report (CLI).

Single-user local tool. Runs the full pipeline (prep -> MD -> analysis) and emits
a self-contained explorer HTML per molecule. MD assumes a local GPU; prep and
analysis are CPU. Run inside the `mdsprep` conda env.

Examples:
    python run.py "O=C(O)c1ccccc1"                 --name aspirin
    python run.py --input molecules.csv --outdir results
    python run.py "CCO" --production-ns 50 --n-frames 500 --force
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ensemble_qsar.cli import orchestrate  # noqa: E402

try:
    from tqdm import tqdm  # optional progress bar
    _progress = lambda it, total: tqdm(it, total=total, unit="mol")  # noqa: E731
except Exception:  # noqa: BLE001
    _progress = None


def _overrides(a: argparse.Namespace) -> dict:
    keys = ("pH", "production_ns", "save_interval_ps", "n_frames",
            "n_clusters", "water_model", "platform", "seed")
    return {k: getattr(a, k) for k in keys if getattr(a, k) is not None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("smiles", nargs="?", help="a single SMILES string")
    ap.add_argument("--name", help="molecule name (default: hash slug of SMILES)")
    ap.add_argument("--input", type=Path, help="CSV with a 'smiles' (+ optional 'name') column")
    ap.add_argument("--outdir", type=Path, default=Path("results"))
    ap.add_argument("--force", action="store_true", help="re-run all stages (wipe molecule dir)")
    # config overrides (default from Prep/MD config)
    ap.add_argument("--pH", type=float)
    ap.add_argument("--production-ns", dest="production_ns", type=float)
    ap.add_argument("--save-interval-ps", dest="save_interval_ps", type=float)
    ap.add_argument("--n-frames", dest="n_frames", type=int, help="frames = production/save; sets save interval")
    ap.add_argument("--n-clusters", dest="n_clusters", type=int)
    ap.add_argument("--water-model", dest="water_model")
    ap.add_argument("--platform", help="CUDA / OpenCL / CPU")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()

    overrides = _overrides(args)
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.input:
        rows = list(csv.DictReader(args.input.open()))
        results = orchestrate.run_many(rows, outdir=args.outdir, overrides=overrides,
                                       force=args.force, progress=_progress)
        sys.exit(0 if any(r.status == "ok" for r in results) else 1)

    if not args.smiles:
        ap.error("provide a SMILES argument or --input CSV")
    res = orchestrate.run_one(args.smiles, name=args.name, outdir=args.outdir,
                              overrides=overrides, force=args.force)
    if res.status == "ok":
        print(f"\nreport: {res.report}")
        sys.exit(0)
    print(f"\nFAILED: {res.error}")
    sys.exit(1)


if __name__ == "__main__":
    main()
