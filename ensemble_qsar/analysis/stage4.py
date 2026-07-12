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

# Reference-free descriptors only — comparable across molecules. RMSD is
# excluded on purpose: it is measured against each molecule's own reference
# structure, so its magnitude is not comparable between molecules.
DESCRIPTORS = ["psa3d", "rg", "intra_hbond", "sasa"]
_UNIT = {"psa3d": "3D-PSA (Å²)", "rg": "Rg (Å)",
         "intra_hbond": "intra H-bonds", "sasa": "SASA (Å²)"}
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


def plot_analysis1(table: pd.DataFrame, out_path: Path, *,
                   descriptors=("psa3d", "rg"), dispersion: str = "std") -> Path:
    """Per-molecule per-frame dispersion vs flexibility. One row per descriptor
    (each reference-free, so comparable across molecules), columns = the two
    flexibility axes (rotatable-bond count and Kier φ). One point = one molecule.
    """
    descriptors = [d for d in descriptors if f"{d}_{dispersion}" in table.columns]
    nrow = len(descriptors)
    fig, axes = plt.subplots(nrow, 2, figsize=(11, 4.2 * nrow), squeeze=False)
    xaxes = (("n_rotatable_bonds", "rotatable bonds"), ("kier_flexibility", "Kier flexibility φ"))
    for r, desc in enumerate(descriptors):
        ycol = f"{desc}_{dispersion}"
        for c, (xcol, xlabel) in enumerate(xaxes):
            ax = axes[r][c]
            for cls in FLEX_ORDER:
                sub = table[table["flex_class"] == cls]
                if len(sub):
                    ax.scatter(sub[xcol], sub[ycol], s=70, alpha=0.85, color=FLEX_COLOR[cls],
                               edgecolor="#333", linewidth=0.5, label=cls)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(f"per-frame {_UNIT[desc]} {dispersion}")
            ax.grid(alpha=0.25)
    axes[0][0].legend(title="flex_class", fontsize=9, framealpha=0.9)
    fig.suptitle(f"Ensemble descriptor dispersion vs flexibility "
                 f"(n={len(table)}, one point = one molecule; exploratory — no model)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return Path(out_path)


def _dir_for_mol_ids(md_root: Path) -> dict:
    return {_mol_id(d): d for d in Path(md_root).iterdir()
            if d.is_dir() and not d.name.startswith("_")}


def select_case_studies(table: pd.DataFrame, *, n_each: int = 2) -> list[str]:
    """Auto-pick the `n_each` most-rigid and `n_each` most-flexible molecules
    (by Kier φ) with a valid static TPSA + label, for the case-study contrast."""
    t = table.dropna(subset=["kier_flexibility", "tpsa"]).sort_values("kier_flexibility")
    return list(t["mol_id"].head(n_each)) + list(t["mol_id"].tail(n_each))


def case_study(md_root: Path, table: pd.DataFrame, *, mol_ids=None, n_each: int = 2) -> list[dict]:
    """Per-frame 3D-PSA distribution + static 2D-TPSA + label for selected molecules."""
    mol_ids = mol_ids or select_case_studies(table, n_each=n_each)
    dirs = _dir_for_mol_ids(md_root)
    row = table.set_index("mol_id")
    cases = []
    for mid in mol_ids:
        if mid not in dirs or mid not in row.index:
            continue
        pf = _per_frame(dirs[mid])
        if pf is None or "psa3d" not in pf.columns:
            continue
        r = row.loc[mid]
        cases.append({
            "mol_id": mid, "flex_class": str(r.get("flex_class")),
            "n_rot": float(r.get("n_rotatable_bonds", float("nan"))),
            "psa3d_frames": pf["psa3d"].to_numpy(float),
            "static_tpsa": float(r.get("tpsa", float("nan"))),
            "label": float(r.get("label", float("nan"))),
        })
    # sort rigid -> flexible for the figure
    return sorted(cases, key=lambda c: c["n_rot"])


def plot_analysis2(cases: list[dict], out_path: Path) -> Path:
    """Distribution (violin) of per-frame 3D-PSA per molecule + static 2D-TPSA
    marker + experimental Caco-2 label annotation. Rigid = narrow (static point
    representative); flexible = wide (single static point loses representativeness).
    """
    n = len(cases)
    fig, ax = plt.subplots(figsize=(9, 1.0 + 0.9 * n))
    pos = list(range(n))
    data = [c["psa3d_frames"] for c in cases]
    vp = ax.violinplot(data, positions=pos, vert=False, showextrema=True, widths=0.8)
    for body, c in zip(vp["bodies"], cases):
        body.set_facecolor(FLEX_COLOR.get(c["flex_class"], "#999"))
        body.set_alpha(0.5)
    # static 2D-TPSA marker + label annotation
    for i, c in enumerate(cases):
        ax.scatter([c["static_tpsa"]], [i], marker="D", s=70, color="#111",
                   zorder=5, label="static 2D-TPSA" if i == 0 else None)
        ax.text(max(c["psa3d_frames"].max(), c["static_tpsa"]) + 3, i,
                f"Caco-2 logPapp = {c['label']:.2f}", va="center", fontsize=8, color="#555")
    ax.set_yticks(pos)
    ax.set_yticklabels([f"{c['mol_id'][:26]}\n({c['flex_class']}, {int(c['n_rot'])} rot)" for c in cases], fontsize=8)
    ax.set_xlabel("3D-PSA (Å²)   —   violin: MD ensemble per-frame distribution")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Case study: static 2D-TPSA vs MD 3D-PSA distribution\n"
                 "(rigid = narrow → static representative; flexible = broad → static loses it)",
                 fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return Path(out_path)
