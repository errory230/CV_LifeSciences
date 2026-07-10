#!/usr/bin/env python3
"""Batch prep driver: run the prep pipeline over many molecules, fail-soft.

Reads a CSV with `mol_id` and `smiles` columns (e.g. a Stage-1
representative_set.csv or the validation subset), preps each molecule, and
writes a `batch_report.csv` / `batch_report.json`. A molecule that fails (or a
peptide whose sequence can't be parsed) is logged and skipped so the batch never
halts on one bad input.

Run inside the `mdsprep` conda env:
    conda run -n mdsprep python scripts/03_prep_batch.py --input data/validation_subset.csv
    conda run -n mdsprep python scripts/03_prep_batch.py \
        --input data/processed/caco2_wang/representative_set.csv --out-root data/prep/caco2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ensemble_qsar.prep.pipeline import PrepConfig, prep_molecule  # noqa: E402


def _read_rows(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open()))
    # tolerate stage-1 ('smiles') and validation ('mol_id'/'-') schemas
    out = []
    for r in rows:
        smi = r.get("smiles") or r.get("SMILES")
        mid = r.get("mol_id") or r.get("Drug_ID") or ""
        if mid in ("", "-"):
            mid = r.get("tier") or smi[:12]
        if smi:
            out.append({"mol_id": mid, "smiles": smi})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out-root", type=Path, default=ROOT / "data" / "prep")
    ap.add_argument("--ph", type=float, default=7.4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--water-model", default="tip3p")
    ap.add_argument("--limit", type=int, default=0, help="cap molecules (0 = all)")
    args = ap.parse_args()

    rows = _read_rows(args.input)
    if args.limit:
        rows = rows[: args.limit]
    cfg = PrepConfig(ph=args.ph, random_seed=args.seed, water_model=args.water_model)
    args.out_root.mkdir(parents=True, exist_ok=True)

    report: list[dict] = []
    n_ok = n_skip = n_fail = 0
    for i, row in enumerate(rows, 1):
        mid, smi = row["mol_id"], row["smiles"]
        rec = {"mol_id": mid, "smiles": smi, "status": "", "reason": "",
               "mol_class": "", "needs_review": None, "net_charge": None,
               "handoff_files": []}
        print(f"[{i}/{len(rows)}] {mid} ... ", end="", flush=True)
        try:
            m = prep_molecule(mid, smi, out_root=args.out_root, config=cfg)
            rec.update(status="ok", mol_class=m.mol_class, needs_review=m.needs_review,
                       net_charge=m.net_charge, handoff_files=m.handoff_files)
            n_ok += 1
            print(f"ok ({m.mol_class}, review={m.needs_review})")
        except NotImplementedError as e:
            rec.update(status="skipped", reason=str(e))
            n_skip += 1
            print(f"skipped ({e})")
        except Exception as e:  # noqa: BLE001 fail-soft by design
            rec.update(status="failed", reason=f"{type(e).__name__}: {e}")
            (args.out_root / "_errors").mkdir(exist_ok=True)
            (args.out_root / "_errors" / f"{i:03d}_{_safe(mid)}.txt").write_text(
                traceback.format_exc())
            n_fail += 1
            print(f"FAILED ({type(e).__name__})")
        report.append(rec)

    (args.out_root / "batch_report.json").write_text(json.dumps(report, indent=2))
    with (args.out_root / "batch_report.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(report[0].keys()))
        w.writeheader()
        for r in report:
            r = dict(r, handoff_files=";".join(r["handoff_files"]))
            w.writerow(r)
    print(f"\n== done: {n_ok} ok, {n_skip} skipped, {n_fail} failed "
          f"-> {args.out_root}/batch_report.csv ==")


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


if __name__ == "__main__":
    main()
