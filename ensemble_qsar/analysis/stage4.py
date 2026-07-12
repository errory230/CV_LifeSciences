"""Stage 4 (exploratory): MD-ensemble feature dispersion vs. flexibility.

This is a DESCRIPTIVE, proof-of-concept analysis — no model is trained and no
predictive accuracy is claimed (molecule count is small). The idea:

  a flexible molecule samples a wide range of 3D-PSA (and other shape-dependent
  descriptors) across its MD ensemble, so representing it by a single static
  value (e.g. 2D TPSA) discards real spread. We treat the *width* of a molecule's
  per-frame descriptor distribution as a proxy for that representational
  uncertainty and ask, purely qualitatively, whether it grows with flexibility.

Reads only already-extracted Stage-3 outputs (`viz_data.json`, or `metrics.csv`
as a fallback) plus the Stage-1 CSV (flexibility + Caco-2 label). CPU only.

Explicitly NOT done here: regression/ML, R^2, per-cluster predictions, or picking
"which cluster is the true experimental value".
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"   # override Colab/IPython inline backend
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DESCRIPTORS = ["psa3d", "rg", "intra_hbond", "sasa", "rmsd"]
FLEX_ORDER = ["rigid", "moderate", "flexible", "very_flexible"]
FLEX_COLOR = {"rigid": "#2c7fb8", "moderate": "#41ab5d",
              "flexible": "#fe9929", "very_flexible": "#d7301f"}


def _mol_id(md_dir: Path) -> str:
    for name in ("viz_data.json", "run_manifest.json", "manifest.json"):
        p = md_dir / name
        if p.exists():
            d = json.loads(p.read_text())
            mid = d.get("meta", {}).get("mol_id") or \
                d.get("prep_manifest", d).get("mol_id")
            if mid:
                return mid
    return md_dir.name


def _per_frame(md_dir: Path) -> pd.DataFrame | None:
    """Per-frame descriptor table for one molecule (viz_data.json, else metrics)."""
    viz = md_dir / "viz_data.json"
    if viz.exists():
        frames = json.loads(viz.read_text()).get("frames", [])
        return pd.DataFrame(frames) if frames else None
    m = md_dir / "analysis" / "metrics.csv"
    if m.exists():
        df = pd.read_csv(m).rename(columns={"psa3d_A2": "psa3d", "rmsd_A": "rmsd"})
        return df
    return None


def _dispersion(v: np.ndarray) -> dict:
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "std": float(v.std()),
            "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
            "range": float(v.max() - v.min())}


def detect_and_tabulate(md_root: Path, stage1_csv: Path, *, burnin_frac: float = 0.0):
    """Return (table, pending). One row per completed molecule with per-frame
    dispersion of each descriptor joined to Stage-1 flexibility + label."""
    md_root = Path(md_root)
    s1 = pd.read_csv(stage1_csv)
    keep = [c for c in ("mol_id", "n_rotatable_bonds", "kier_flexibility",
                        "flex_class", "label", "tpsa", "mw") if c in s1.columns]
    s1 = s1[keep]

    rows, pending = [], []
    for d in sorted(p for p in md_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        pf = _per_frame(d)
        if pf is None or len(pf) == 0:
            pending.append(d.name)
            continue
        if burnin_frac > 0:
            pf = pf.iloc[int(len(pf) * burnin_frac):]
        row = {"mol_id": _mol_id(d), "n_frames": int(len(pf))}
        for desc in DESCRIPTORS:
            if desc in pf.columns:
                for k, val in _dispersion(pf[desc]).items():
                    row[f"{desc}_{k}"] = val
        rows.append(row)

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.merge(s1, on="mol_id", how="left")
        table["flex_class"] = pd.Categorical(table.get("flex_class"), FLEX_ORDER, ordered=True)
        table = table.sort_values(["flex_class", "n_rotatable_bonds"], na_position="last")
    return table, pending


def plot_analysis1(table: pd.DataFrame, out_path: Path, *, descriptor: str = "psa3d",
                   dispersion: str = "std") -> Path:
    """Scatter: per-frame descriptor dispersion vs flexibility (n_rot & Kier)."""
    ycol = f"{descriptor}_{dispersion}"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, xcol, xlabel in ((axes[0], "n_rotatable_bonds", "rotatable bonds"),
                             (axes[1], "kier_flexibility", "Kier flexibility φ")):
        for cls in FLEX_ORDER:
            sub = table[table["flex_class"] == cls]
            if len(sub):
                ax.scatter(sub[xcol], sub[ycol], s=70, alpha=0.85,
                           color=FLEX_COLOR[cls], edgecolor="#333", linewidth=0.5, label=cls)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"per-frame {descriptor.upper()} {dispersion}  (Å² spread)")
        ax.grid(alpha=0.25)
    axes[0].legend(title="flex_class", fontsize=9, framealpha=0.9)
    fig.suptitle(f"Ensemble {descriptor.upper()} dispersion grows with flexibility "
                 f"(n={len(table)}, exploratory — no model)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return Path(out_path)
