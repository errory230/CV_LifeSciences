"""Peptide route: build with a protein force field (ff19SB) via tleap.

Peptides are parameterized from their residue *sequence* using the protein force
field's library residues (library charges, per spec) rather than AM1-BCC. The
sequence is parsed from a hyphenated peptide name (e.g.
"Ac-Trp-Ala-Gly-Gly-Lys-Ala-NH2"); automatic SMILES->sequence perception for
unnamed peptides is deliberately out of scope for this stage and such inputs are
flagged for review instead of guessed.

At the default pH (7.4) the standard ff19SB residue names already encode the
physiological ionization (LYS+, ARG+, ASP-, GLU-, HIS=HIE neutral, CYS neutral);
non-default pH would require residue-name swaps (ASH/GLH/LYN/HIP) and is noted as
future work. A built peptide's molecular weight is cross-checked against the
input SMILES so a name/structure mismatch is caught, not silently parameterized.

Cyclic (head-to-tail) peptides get an explicit backbone bond in the tleap recipe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors

from ._util import run_cmd

# N- and C-terminal caps -> Amber residue names.
_CAPS = {
    "ac": "ACE", "acetyl": "ACE",
    "nh2": "NHE", "nh2cap": "NHE", "nhe": "NHE",
    "nme": "NME", "nme2": "NME", "me": "NME",
}
# Standard amino acids (3-letter). HIS defaults to the neutral HIE tautomer.
_AA3 = {
    "ala": "ALA", "arg": "ARG", "asn": "ASN", "asp": "ASP", "cys": "CYS",
    "gln": "GLN", "glu": "GLU", "gly": "GLY", "his": "HIE", "ile": "ILE",
    "leu": "LEU", "lys": "LYS", "met": "MET", "phe": "PHE", "pro": "PRO",
    "ser": "SER", "thr": "THR", "trp": "TRP", "tyr": "TYR", "val": "VAL",
}

_LEAP_PEPTIDE = """source leaprc.protein.ff19SB
pep = sequence {{ {sequence} }}
{cyclic_bond}saveamberparm pep {prmtop} {rst7}
savepdb pep {pdb}
quit
"""

_LEAP_SOLVATE = """# Run on the GPU host to build the solvated peptide system.
source leaprc.protein.ff19SB
source leaprc.water.{water_leaprc}
pep = sequence {{ {sequence} }}
{cyclic_bond}solvateBox pep {water_box} {padding}
addions pep {counterion} 0
saveamberparm pep peptide_solv.prmtop peptide_solv.rst7
savepdb pep peptide_solv.pdb
quit
"""

_WATER_LEAPRC = {"tip3p": ("tip3p", "TIP3PBOX"), "tip4pew": ("tip4pew", "TIP4PEWBOX")}


@dataclass
class SequenceResult:
    residues: list[str]
    ok: bool
    notes: list[str] = field(default_factory=list)


def sequence_from_name(name: str) -> SequenceResult:
    """Parse a hyphenated peptide name into an Amber residue sequence."""
    notes: list[str] = []
    tokens = [t for t in name.replace("_", "-").split("-") if t]
    if len(tokens) < 2:
        return SequenceResult([], False, ["name is not a hyphenated peptide"])

    residues: list[str] = []
    for i, tok in enumerate(tokens):
        key = tok.strip().lower()
        is_terminal = i == 0 or i == len(tokens) - 1
        if is_terminal and key in _CAPS:
            residues.append(_CAPS[key])
        elif key in _AA3:
            if key == "his":
                notes.append("HIS -> HIE (neutral tautomer assumed)")
            residues.append(_AA3[key])
        else:
            return SequenceResult([], False, [f"unrecognized residue/cap: {tok!r}"])
    return SequenceResult(residues, True, notes)


@dataclass
class PeptideResult:
    gas_prmtop: Path
    gas_rst7: Path
    capped_pdb: Path
    solvate_recipe: Path
    residues: list[str]
    net_charge: int
    ok: bool
    messages: list[str] = field(default_factory=list)


def build_peptide(
    residues: list[str],
    *,
    out_dir: Path,
    reference_smiles: str,
    cyclic: bool = False,
    water_model: str = "tip3p",
    box_padding: float = 12.0,
    mw_tol: float = 5.0,
) -> PeptideResult:
    """Build a peptide with ff19SB via tleap; validate MW against the SMILES."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gas_prmtop = out_dir / "peptide_gas.prmtop"
    gas_rst7 = out_dir / "peptide_gas.rst7"
    capped_pdb = out_dir / "peptide_capped.pdb"
    messages: list[str] = []
    seq_str = " ".join(residues)
    n = len(residues)
    # Head-to-tail bond for cyclic peptides (first residue N to last residue C).
    cyclic_bond = f"bond pep.1.N pep.{n}.C\n" if cyclic else ""

    leap_gas = out_dir / "leap_peptide.in"
    leap_gas.write_text(_LEAP_PEPTIDE.format(
        sequence=seq_str, cyclic_bond=cyclic_bond,
        prmtop=gas_prmtop.name, rst7=gas_rst7.name, pdb=capped_pdb.name))
    tl = run_cmd(["tleap", "-f", leap_gas.name], cwd=out_dir, log_path=out_dir / "tleap_peptide.log")

    ok = gas_prmtop.exists() and gas_prmtop.stat().st_size > 0
    net_charge = 0
    if ok:
        import parmed

        p = parmed.load_file(str(gas_prmtop))
        net_charge = round(sum(a.charge for a in p.atoms))
        built_mw = sum(a.mass for a in p.atoms)
        ref = Chem.MolFromSmiles(reference_smiles)
        ref_mw = Descriptors.MolWt(ref) if ref else 0.0
        if ref_mw and abs(built_mw - ref_mw) > mw_tol:
            ok = False
            messages.append(
                f"MW mismatch: built {built_mw:.1f} vs SMILES {ref_mw:.1f} "
                f"(> {mw_tol}); name/structure inconsistent")
    else:
        messages.append("tleap peptide build failed")

    # Solvation recipe for the GPU host.
    leaprc, boxword = _WATER_LEAPRC.get(water_model, _WATER_LEAPRC["tip3p"])
    counterion = "Na+" if net_charge < 0 else "Cl-"
    solvate = out_dir / "leap_solvate.in"
    solvate.write_text(_LEAP_SOLVATE.format(
        water_leaprc=leaprc, sequence=seq_str, cyclic_bond=cyclic_bond,
        water_box=boxword, padding=box_padding, counterion=counterion))

    return PeptideResult(gas_prmtop, gas_rst7, capped_pdb, solvate,
                         residues, net_charge, ok, messages)
