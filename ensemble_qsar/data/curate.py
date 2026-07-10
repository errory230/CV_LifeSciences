"""Structure curation for QSAR-ready datasets.

Raw property files contain salts, mixtures, charged forms, duplicates and the
occasional unparseable SMILES. Before any conformer generation or MD we bring
every record to a single, neutral, canonical parent structure and collapse
duplicates. This is standard cheminformatics hygiene (cf. Fourches et al. 2010,
"Trust, but Verify") and it matters doubly here: the same 3D-ensemble pipeline
must be applied to a *well-defined* 2D structure per record.

The public entry point is `curate_frame`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

# Elements we accept as "organic, MD-tractable" small molecules. Anything with
# a metal / exotic element is dropped and reported rather than silently kept.
_ORGANIC_ELEMENTS = {
    "H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I",
}


@dataclass
class CurationReport:
    """Counts of what happened, so a run is auditable, not a black box."""

    n_input: int = 0
    n_unparseable: int = 0
    n_no_carbon: int = 0
    n_disallowed_element: int = 0
    n_standardize_failed: int = 0
    n_duplicate_records: int = 0
    n_output: int = 0

    def as_dict(self) -> dict[str, int]:
        return self.__dict__.copy()


def _largest_fragment_parent(mol: Chem.Mol) -> Chem.Mol | None:
    """Clean up, keep the largest organic fragment, and neutralize it."""
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)  # largest fragment, salt-stripped
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def _is_organic(mol: Chem.Mol) -> bool:
    symbols = {a.GetSymbol() for a in mol.GetAtoms()}
    return symbols.issubset(_ORGANIC_ELEMENTS)


def _has_carbon(mol: Chem.Mol) -> bool:
    return any(a.GetAtomicNum() == 6 for a in mol.GetAtoms())


def standardize_smiles(smiles: str) -> tuple[str | None, str | None, str]:
    """Return `(canonical_smiles, inchikey, status)` for one raw SMILES.

    status is one of: "ok", "unparseable", "no_carbon",
    "disallowed_element", "standardize_failed".
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None, None, "unparseable"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, "unparseable"

    parent = _largest_fragment_parent(mol)
    if parent is None:
        return None, None, "standardize_failed"
    if not _has_carbon(parent):
        return None, None, "no_carbon"
    if not _is_organic(parent):
        return None, None, "disallowed_element"

    try:
        can = Chem.MolToSmiles(parent)
        key = inchi.MolToInchiKey(parent)
    except Exception:
        return None, None, "standardize_failed"
    if not key:
        return None, None, "standardize_failed"
    return can, key, "ok"


def curate_frame(
    df: pd.DataFrame,
    *,
    smiles_col: str,
    label_col: str,
    id_col: str | None = None,
    dedup: str = "median",
) -> tuple[pd.DataFrame, CurationReport]:
    """Curate a raw property table into a QSAR-ready frame.

    Output columns: `mol_id, smiles, inchikey, label, n_measurements`.
    Duplicate structures (same InChIKey) are collapsed; their labels are
    aggregated by `dedup` ("median" or "mean") and the spread recorded.
    """
    rep = CurationReport(n_input=len(df))
    rows: list[dict] = []

    for _, row in df.iterrows():
        can, key, status = standardize_smiles(row[smiles_col])
        if status != "ok":
            setattr(rep, f"n_{status}", getattr(rep, f"n_{status}") + 1)
            continue
        label = pd.to_numeric(row[label_col], errors="coerce")
        if not np.isfinite(label):
            rep.n_unparseable += 1
            continue
        mol_id = str(row[id_col]) if id_col and id_col in df.columns else key
        rows.append(
            {"mol_id": mol_id, "smiles": can, "inchikey": key, "label": float(label)}
        )

    if not rows:
        return (
            pd.DataFrame(columns=["mol_id", "smiles", "inchikey", "label", "n_measurements"]),
            rep,
        )

    raw = pd.DataFrame(rows)
    agg = "median" if dedup == "median" else "mean"
    grouped = raw.groupby("inchikey", as_index=False).agg(
        mol_id=("mol_id", "first"),
        smiles=("smiles", "first"),
        label=("label", agg),
        label_std=("label", "std"),
        n_measurements=("label", "size"),
    )
    grouped["label_std"] = grouped["label_std"].fillna(0.0)
    rep.n_duplicate_records = len(raw) - len(grouped)
    rep.n_output = len(grouped)

    return (
        grouped[["mol_id", "smiles", "inchikey", "label", "label_std", "n_measurements"]],
        rep,
    )
