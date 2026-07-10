"""Rule-based molecule classification to route parameterization.

The prep pipeline parameterizes peptides and general small molecules by
different routes (protein force field vs GAFF2/AM1-BCC). This module decides the
route from the 2D structure with transparent RDKit/SMARTS rules and, crucially,
flags ambiguous cases for human review with the evidence that drove the call.

Decision (see `classify_molecule`):
  * Count backbone residues via the N-Cα-C(=O) motif (`_CALPHA`), which matches
    both substituted α-carbons and glycine.
  * >= `PEPTIDE_MIN_RESIDUES` chained residues  -> "peptide" route.
  * otherwise                                    -> "small_molecule" route.

Review flags (route is still chosen automatically; `needs_review` just marks it
for a human to confirm):
  * macrocycle              -- largest ring >= `MACROCYCLE_MIN_RING`; fragile
                               ring topology (head-to-tail bond, macrolactone).
  * ester in a peptide-like -- possible depsipeptide (ester replaces an amide).
  * N-methylated amide      -- non-standard residue; protein FF libraries lack it.
  * exactly 2 residues      -- borderline dipeptide vs. amide-bearing small mol.
  * carbohydrate-like       -- GLYCAM is preferable to GAFF for sugars.

These same rules are documented in the project README.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rdkit import Chem

# --- tunable thresholds -----------------------------------------------------
PEPTIDE_MIN_RESIDUES = 3
MACROCYCLE_MIN_RING = 12

# --- SMARTS patterns (compiled once) ----------------------------------------
# N-Cα-C(=O): one backbone residue unit; H1 covers substituted Cα, H2 glycine.
_CALPHA = Chem.MolFromSmarts("[NX3][CX4;H1,H2][CX3](=O)")
# Generic amide bond (backbone peptide bond is a subset).
_AMIDE = Chem.MolFromSmarts("[NX3][CX3](=O)")
# Carbonyl ester C(=O)-O-C (excludes carboxylic acid/carboxylate).
_ESTER = Chem.MolFromSmarts("[CX3](=O)[OX2H0][#6]")
# N-methyl amide: carbonyl N bearing a methyl.
_NMETHYL_AMIDE = Chem.MolFromSmarts("[CX3](=O)[NX3]([CH3])")
# Sugar ring oxygen inside a 5/6-membered ring (pyranose/furanose backbone).
_SUGAR_RING_O = Chem.MolFromSmarts("[#6;R][OX2;R][#6;R]")
_HYDROXYL = Chem.MolFromSmarts("[OX2H]")


@dataclass
class ClassificationResult:
    mol_class: str  # "peptide" | "small_molecule"
    needs_review: bool
    n_residues: int
    n_amide_bonds: int
    largest_ring: int
    is_macrocycle: bool
    has_ester: bool
    has_nmethyl_amide: bool
    is_carbohydrate_like: bool
    net_formal_charge: int
    evidence: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _largest_ring(mol: Chem.Mol) -> int:
    return max((len(r) for r in mol.GetRingInfo().AtomRings()), default=0)


def _n_matches(mol: Chem.Mol, patt: Chem.Mol) -> int:
    if patt is None:
        return 0
    return len(mol.GetSubstructMatches(patt, uniquify=True))


def _looks_like_carbohydrate(mol: Chem.Mol) -> bool:
    """Heuristic: >=2 sugar-type ring oxygens and many hydroxyls."""
    n_sugar_o = _n_matches(mol, _SUGAR_RING_O)
    n_oh = _n_matches(mol, _HYDROXYL)
    return n_sugar_o >= 2 and n_oh >= 3


def classify_molecule(smiles: str) -> ClassificationResult:
    """Classify a curated SMILES into a parameterization route."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES: {smiles!r}")

    n_residues = _n_matches(mol, _CALPHA)
    n_amide = _n_matches(mol, _AMIDE)
    largest = _largest_ring(mol)
    is_macro = largest >= MACROCYCLE_MIN_RING
    has_ester = _n_matches(mol, _ESTER) > 0
    has_nmethyl = _n_matches(mol, _NMETHYL_AMIDE) > 0
    is_carb = _looks_like_carbohydrate(mol)
    net_charge = Chem.GetFormalCharge(mol)

    evidence: list[str] = [
        f"residues(N-Ca-C=O)={n_residues}",
        f"amide_bonds={n_amide}",
        f"largest_ring={largest}",
    ]

    is_peptide = n_residues >= PEPTIDE_MIN_RESIDUES
    mol_class = "peptide" if is_peptide else "small_molecule"
    evidence.append(
        f"class={mol_class} (rule: residues>={PEPTIDE_MIN_RESIDUES} -> peptide)"
    )

    needs_review = False
    if is_macro:
        needs_review = True
        evidence.append(f"REVIEW: macrocycle (ring>={MACROCYCLE_MIN_RING}); "
                        "explicit ring-closure topology needed")
    if has_ester and n_residues >= 2:
        needs_review = True
        evidence.append("REVIEW: ester + peptide-like -> possible depsipeptide")
    if has_nmethyl and is_peptide:
        needs_review = True
        evidence.append("REVIEW: N-methyl amide -> non-standard residue")
    if n_residues == 2:
        needs_review = True
        evidence.append("REVIEW: 2 residues -> borderline dipeptide/small-molecule")
    if is_carb:
        needs_review = True
        evidence.append("REVIEW: carbohydrate-like -> GLYCAM recommended over GAFF")

    return ClassificationResult(
        mol_class=mol_class,
        needs_review=needs_review,
        n_residues=n_residues,
        n_amide_bonds=n_amide,
        largest_ring=largest,
        is_macrocycle=is_macro,
        has_ester=has_ester,
        has_nmethyl_amide=has_nmethyl,
        is_carbohydrate_like=is_carb,
        net_formal_charge=net_charge,
        evidence=evidence,
    )
