"""Representative-set selection for the MD stage.

Running solvent MD + conformer extraction on every molecule is infeasible, and
also unnecessary: the PoC only needs a *representative* subset. We choose it to
make the ensemble-vs-static hypothesis testable, which requires two things at
once:

  1. Flexibility coverage. We stratify by rotatable-bond class and deliberately
     over-sample the flexible / very-flexible tail (where ensembles should
     matter) while keeping a rigid control group (where they should not). The
     contrast between strata is the actual PoC signal.

  2. Chemical diversity within each stratum. Using a MaxMin picker on Morgan
     fingerprints we spread the picks across chemical space instead of grabbing
     near-duplicates, so MD compute is not wasted on redundant scaffolds.

The output is a table of selected molecules with the reason each was picked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.SimDivFilters import rdSimDivPickers

from .descriptors import FLEX_LABELS

# Default share of the MD budget per flexibility stratum. Deliberately weighted
# toward flexible molecules; rigid molecules act as a negative control.
DEFAULT_STRATUM_WEIGHTS = {
    "rigid": 0.15,
    "moderate": 0.25,
    "flexible": 0.30,
    "very_flexible": 0.30,
}


def _fingerprints(smiles: list[str]) -> list[DataStructs.ExplicitBitVect]:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        fps.append(gen.GetFingerprint(mol))
    return fps


def _maxmin_pick(
    fps: list[DataStructs.ExplicitBitVect], k: int, seed: int
) -> list[int]:
    """Pick `k` maximally-diverse indices from `fps` via MaxMin."""
    n = len(fps)
    if k >= n:
        return list(range(n))

    def dist(i: int, j: int) -> float:
        return 1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j])

    picker = rdSimDivPickers.MaxMinPicker()
    picked = picker.LazyPick(dist, n, k, seed=seed)
    return list(picked)


def select_representative_set(
    df: pd.DataFrame,
    *,
    n_select: int,
    smiles_col: str = "smiles",
    flex_col: str = "flex_class",
    weights: dict[str, float] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Select `n_select` molecules stratified by flexibility, diverse within.

    Returns the selected rows of `df` with two added columns:
      * `selection_stratum` -- the flexibility class it was drawn from.
      * `selection_reason`  -- short human-readable rationale.
    """
    weights = weights or DEFAULT_STRATUM_WEIGHTS
    n_select = min(n_select, len(df))

    # Allocate the budget across strata that are actually present, renormalising.
    present = {c: (df[flex_col] == c).sum() for c in FLEX_LABELS}
    active = {c: w for c, w in weights.items() if present.get(c, 0) > 0}
    wsum = sum(active.values()) or 1.0
    alloc = {c: int(round(n_select * w / wsum)) for c, w in active.items()}
    # Cap allocation at availability and fix rounding drift.
    for c in alloc:
        alloc[c] = min(alloc[c], present[c])
    _repair_allocation(alloc, present, n_select)

    picks: list[pd.DataFrame] = []
    for cls, k in alloc.items():
        if k <= 0:
            continue
        sub = df[df[flex_col] == cls].reset_index()  # keep original index
        idx = _maxmin_pick(_fingerprints(sub[smiles_col].tolist()), k, seed)
        chosen = sub.iloc[idx].copy()
        chosen["selection_stratum"] = cls
        chosen["selection_reason"] = f"diverse pick from '{cls}' stratum"
        picks.append(chosen)

    out = pd.concat(picks, ignore_index=True)
    out = out.drop(columns=["index"])
    return out


def _repair_allocation(
    alloc: dict[str, int], present: dict[str, int], target: int
) -> None:
    """Adjust `alloc` in place so it sums to `target` without exceeding supply."""
    def total() -> int:
        return sum(alloc.values())

    # Add where there is headroom.
    while total() < target:
        headroom = {c: present[c] - alloc[c] for c in alloc if present[c] - alloc[c] > 0}
        if not headroom:
            break
        c = max(headroom, key=headroom.get)
        alloc[c] += 1
    # Remove excess from the largest allocations.
    while total() > target:
        c = max(alloc, key=alloc.get)
        if alloc[c] == 0:
            break
        alloc[c] -= 1
