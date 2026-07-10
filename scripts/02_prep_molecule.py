#!/usr/bin/env python3
"""Prep a single molecule (dev/T-tier driver for Stage 2a).

Run inside the `mdsprep` conda env:
    conda run -n mdsprep python scripts/02_prep_molecule.py --smiles c1ccncc1 --mol-id T1_smoke
    conda run -n mdsprep python scripts/02_prep_molecule.py --from-validation T1_smoke
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ensemble_qsar.prep.pipeline import PrepConfig, prep_molecule  # noqa: E402

VALIDATION = ROOT / "data" / "validation_subset.csv"
OUT_ROOT = ROOT / "data" / "prep"


def _lookup_validation(key: str) -> tuple[str, str]:
    for row in csv.DictReader(VALIDATION.open()):
        mid = row["mol_id"] if row["mol_id"] != "-" else row["tier"]
        if key in (row["tier"], row["mol_id"], mid):
            return mid, row["smiles"]
    raise SystemExit(f"'{key}' not found in {VALIDATION}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smiles")
    ap.add_argument("--mol-id")
    ap.add_argument("--from-validation", help="tier or mol_id from validation_subset.csv")
    ap.add_argument("--ph", type=float, default=7.4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--water-model", default="tip3p")
    args = ap.parse_args()

    if args.from_validation:
        mol_id, smiles = _lookup_validation(args.from_validation)
    elif args.smiles and args.mol_id:
        mol_id, smiles = args.mol_id, args.smiles
    else:
        raise SystemExit("provide --from-validation, or both --smiles and --mol-id")

    cfg = PrepConfig(ph=args.ph, random_seed=args.seed, water_model=args.water_model)
    print(f"=== prep {mol_id} ===\n  SMILES: {smiles}")
    m = prep_molecule(mol_id, smiles, out_root=OUT_ROOT, config=cfg)
    print(f"  class: {m.mol_class}  review: {m.needs_review}  net_charge: {m.net_charge}")
    print(f"  protonation@pH{m.ph}: {m.protonation_state}")
    for s in m.steps:
        print(f"    [{s.status:7s}] {s.name}  ({s.seconds}s)")
    print(f"  handoff files -> {m.path}")
    for f in m.handoff_files:
        print(f"    - {f}")


if __name__ == "__main__":
    main()
