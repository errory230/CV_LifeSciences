#!/usr/bin/env python3
"""Stage 1 pipeline: raw property benchmark -> curated, annotated, selected.

For each requested dataset this script:
  1. downloads the raw file from its pinned mirror (cached, checksummed),
  2. curates structures (parent/neutral/canonical, dedup),
  3. computes physicochemical + flexibility descriptors,
  4. selects a flexibility-stratified, chemically-diverse representative set
     for the molecular-dynamics stage,
  5. writes everything to data/processed/ plus a JSON run summary.

Usage:
    python scripts/01_build_poc_dataset.py                 # all datasets
    python scripts/01_build_poc_dataset.py --datasets caco2_wang esol
    python scripts/01_build_poc_dataset.py --n-select 80 --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ensemble_qsar.data import curate, descriptors, select, sources  # noqa: E402

PROCESSED = ROOT / "data" / "processed"


def build_one(name: str, *, n_select: int, force: bool, seed: int) -> dict:
    ds = sources.resolve(name)
    print(f"\n=== {ds.name}  ({ds.endpoint}, {ds.label_unit}) ===")

    raw_path = sources.fetch(ds, force=force)
    raw = pd.read_csv(raw_path, sep=ds.sep)
    print(f"  raw rows: {len(raw)}")

    curated, report = curate.curate_frame(
        raw, smiles_col=ds.smiles_col, label_col=ds.label_col, id_col=ds.id_col
    )
    print(f"  curated molecules: {report.n_output}  (report: {report.as_dict()})")

    annotated = descriptors.add_descriptors(curated)
    outdir = PROCESSED / ds.name
    outdir.mkdir(parents=True, exist_ok=True)
    annotated.to_csv(outdir / "curated.csv", index=False)
    annotated.to_parquet(outdir / "curated.parquet", index=False)

    rep_set = select.select_representative_set(
        annotated, n_select=n_select, seed=seed
    )
    rep_set.to_csv(outdir / "representative_set.csv", index=False)
    print(
        f"  representative set: {len(rep_set)} molecules "
        f"-> {outdir / 'representative_set.csv'}"
    )

    flex_counts = annotated["flex_class"].value_counts().to_dict()
    rep_flex_counts = rep_set["selection_stratum"].value_counts().to_dict()
    summary = {
        "dataset": ds.name,
        "endpoint": ds.endpoint,
        "task": ds.task,
        "label_unit": ds.label_unit,
        "source_url": ds.url,
        "provenance": ds.note,
        "curation": report.as_dict(),
        "n_curated": int(report.n_output),
        "flex_class_counts": {k: int(v) for k, v in flex_counts.items()},
        "n_beyond_ro5": int(annotated["beyond_ro5"].sum()),
        "n_macrocycle": int(annotated["is_macrocycle"].sum()),
        "label_min": float(annotated["label"].min()),
        "label_max": float(annotated["label"].max()),
        "n_representative": int(len(rep_set)),
        "representative_flex_counts": {k: int(v) for k, v in rep_flex_counts.items()},
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=sorted(sources.REGISTRY),
        help="dataset names/aliases (default: all registered)",
    )
    ap.add_argument(
        "--n-select",
        type=int,
        default=100,
        help="molecules to select per dataset for the MD stage (default 100)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="ignore download cache")
    args = ap.parse_args()

    PROCESSED.mkdir(parents=True, exist_ok=True)
    summaries = [
        build_one(name, n_select=args.n_select, force=args.force, seed=args.seed)
        for name in args.datasets
    ]
    (PROCESSED / "run_summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nWrote run summary -> {PROCESSED / 'run_summary.json'}")


if __name__ == "__main__":
    main()
