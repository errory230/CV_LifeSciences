"""Step: assign the dominant protonation (micro)state at a target pH.

For small molecules we use Dimorphite-DL, which enumerates protonation states
whose transitions fall inside a pH window. We query a tight window centred on
the target pH and take the most-probable microstate as the structure to
parameterize, recording every enumerated variant and the tool version for
provenance. Protonation is resolved on the 2D SMILES *before* 3D embedding,
because adding/removing protons changes the atom set and net charge that the
conformer and charge steps depend on.

Peptide protonation at pH is handled separately in the peptide route
(pdb2pqr/propka) and is not covered here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rdkit import Chem

# Chemically implausible amide ionizations near pH 7 that Dimorphite can emit:
#   - deprotonated amide N ("amidate"), amide pKa ~17
#   - protonated amide N (amides protonate on O, and only under strong acid)
# Any variant containing either is rejected as a safety guard.
_IMPLAUSIBLE = (
    Chem.MolFromSmarts("[NX2-][CX3]=O"),   # deprotonated amide
    Chem.MolFromSmarts("[NX4+][CX3]=O"),   # protonated amide
)


@dataclass
class ProtonationResult:
    smiles: str  # dominant microstate (canonical)
    net_charge: int
    ph: float
    tool: str
    all_variants: list[str] = field(default_factory=list)
    note: str = ""


def _canonical(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else smi


def _is_plausible(smi: str) -> bool:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    return not any(mol.HasSubstructMatch(p) for p in _IMPLAUSIBLE)


def protonate_small_molecule(
    smiles: str,
    *,
    ph: float = 7.4,
    precision: float = 1.0,
) -> ProtonationResult:
    """Return the dominant protonation microstate at `ph`.

    Queries Dimorphite-DL at a single pH point (`ph_min == ph_max == ph`) so it
    reports the dominant state rather than every state across a window — a
    widened window makes it emit implausible microstates (e.g. deprotonated
    amides). Any residual implausible variant is filtered out as a guard.
    """
    from dimorphite_dl import protonate_smiles  # noqa: PLC0415 (env-specific)

    try:
        import dimorphite_dl

        version = getattr(dimorphite_dl, "__version__", "unknown")
    except Exception:
        version = "unknown"

    variants = protonate_smiles(smiles, ph_min=ph, ph_max=ph, precision=precision,
                                max_variants=16)
    canon = [_canonical(v) for v in variants]
    plausible = [v for v in canon if _is_plausible(v)]

    if not canon:
        dominant = _canonical(smiles)
        note = "no ionizable variants; input structure kept"
    elif not plausible:
        dominant = _canonical(smiles)
        note = "all enumerated variants implausible (amidate); input kept"
    else:
        dominant = plausible[0]
        n_drop = len(canon) - len(plausible)
        note = (f"{len(canon)} microstate(s); {n_drop} implausible dropped; "
                "dominant selected") if len(canon) > 1 else "single microstate"

    mol = Chem.MolFromSmiles(dominant)
    net_charge = Chem.GetFormalCharge(mol) if mol else 0

    return ProtonationResult(
        smiles=dominant,
        net_charge=net_charge,
        ph=ph,
        tool=f"dimorphite-dl {version}",
        all_variants=canon,
        note=note,
    )
