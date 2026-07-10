"""Physicochemical and conformational-flexibility descriptors.

The thesis of this project is that *flexible* molecules sample many
conformations in solution, so a single static structure is a poor QSAR input
for them, whereas rigid molecules are well described by one conformer. To test
that thesis we must know, per molecule, *how flexible it is*. These descriptors
quantify flexibility (and general drug-likeness / beyond-rule-of-5 character) so
that the representative-set selection and the later ensemble-vs-static analysis
can be stratified by flexibility.

Key flexibility signals:
  * n_rotatable_bonds  -- dominant conformational degrees of freedom.
  * kier_flexibility   -- Kier & Hall phi index (kappa1*kappa2/heavy_atoms);
                          a shape-based flexibility measure independent of size.
  * largest_ring / is_macrocycle -- macrocycles (>=12-membered) are a canonical
                          "beyond rule of 5" flexible class (e.g. cyclic peptides).
  * frac_rotatable, fsp3 -- normalised flexibility / 3D character.
"""

from __future__ import annotations

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, GraphDescriptors, rdMolDescriptors

DESCRIPTOR_COLUMNS = [
    "mw",
    "heavy_atoms",
    "n_rotatable_bonds",
    "frac_rotatable",
    "kier_flexibility",
    "n_rings",
    "n_aromatic_rings",
    "n_aliphatic_rings",
    "largest_ring",
    "is_macrocycle",
    "fsp3",
    "tpsa",
    "clogp",
    "n_hbd",
    "n_hba",
    "n_lipinski_violations",
    "beyond_ro5",
]


def _largest_ring(mol: Chem.Mol) -> int:
    rings = mol.GetRingInfo().AtomRings()
    return max((len(r) for r in rings), default=0)


def compute_descriptors(smiles: str) -> dict[str, float]:
    """Compute the descriptor block for one canonical SMILES.

    Assumes `smiles` is already curated (parseable, neutral parent).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {c: float("nan") for c in DESCRIPTOR_COLUMNS}

    heavy = mol.GetNumHeavyAtoms()
    n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    mw = Descriptors.MolWt(mol)
    clogp = Crippen.MolLogP(mol)
    n_hbd = rdMolDescriptors.CalcNumHBD(mol)
    n_hba = rdMolDescriptors.CalcNumHBA(mol)

    # Kier & Hall molecular flexibility index phi = kappa1 * kappa2 / n_atoms.
    kappa1 = GraphDescriptors.Kappa1(mol)
    kappa2 = GraphDescriptors.Kappa2(mol)
    kier = (kappa1 * kappa2 / heavy) if heavy else 0.0

    largest = _largest_ring(mol)
    # Classic Lipinski rule-of-5 violation count.
    violations = sum(
        [mw > 500, clogp > 5, n_hbd > 5, n_hba > 10]
    )

    return {
        "mw": round(mw, 3),
        "heavy_atoms": heavy,
        "n_rotatable_bonds": n_rot,
        "frac_rotatable": round(n_rot / heavy, 4) if heavy else 0.0,
        "kier_flexibility": round(kier, 4),
        "n_rings": rdMolDescriptors.CalcNumRings(mol),
        "n_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "n_aliphatic_rings": rdMolDescriptors.CalcNumAliphaticRings(mol),
        "largest_ring": largest,
        "is_macrocycle": int(largest >= 12),
        "fsp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 4),
        "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 3),
        "clogp": round(clogp, 3),
        "n_hbd": n_hbd,
        "n_hba": n_hba,
        "n_lipinski_violations": int(violations),
        "beyond_ro5": int(mw > 500 or violations >= 2),
    }


# Rotatable-bond bins used throughout the project to stratify "flexibility".
FLEX_BINS = [(-1, 2), (2, 5), (5, 9), (9, float("inf"))]
FLEX_LABELS = ["rigid", "moderate", "flexible", "very_flexible"]


def flexibility_class(n_rotatable_bonds: int) -> str:
    """Map a rotatable-bond count to a coarse flexibility class."""
    for (lo, hi), label in zip(FLEX_BINS, FLEX_LABELS):
        if lo < n_rotatable_bonds <= hi:
            return label
    return FLEX_LABELS[-1]


def add_descriptors(df: pd.DataFrame, *, smiles_col: str = "smiles") -> pd.DataFrame:
    """Return `df` with the descriptor block and `flex_class` appended."""
    desc = df[smiles_col].map(compute_descriptors).apply(pd.Series)
    out = pd.concat([df.reset_index(drop=True), desc], axis=1)
    out["flex_class"] = out["n_rotatable_bonds"].map(flexibility_class)
    return out
