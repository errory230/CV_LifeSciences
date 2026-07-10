"""Step 1: solvation via the prep-provided tleap recipe (leap_solvate.in).

The prep stage emitted a `leap_solvate.in` that loads the dry parameters and
builds a solvated, neutralized box. We run it verbatim with `tleap` (from
AmberTools), so the solvated system uses exactly the prep AM1-BCC charges /
GAFF2 (or ff19SB) parameters. The planned water model in the prep manifest is
checked against the recipe and any mismatch is logged.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SolvationResult:
    prmtop: Path
    rst7: Path
    log: Path
    ok: bool
    warnings: list[str] = field(default_factory=list)


# input files the recipe may reference, copied next to it before running.
_MAYBE_INPUTS = ["ligand.mol2", "ligand.frcmod", "peptide_capped.pdb"]


def _check_water_consistency(recipe_text: str, planned: str) -> list[str]:
    m = re.search(r"leaprc\.water\.(\w+)", recipe_text)
    if not m:
        return ["could not find a water model in leap_solvate.in"]
    used = m.group(1)
    if planned and used.lower() != planned.lower():
        return [f"water-model mismatch: recipe uses {used}, prep planned {planned}"]
    return []


def solvate(mol, out_dir: Path, *, planned_water_model: str = "") -> SolvationResult:
    """Run leap_solvate.in in `out_dir/solvation`; return solvated prmtop/rst7.

    Idempotent: if a solvated prmtop already exists it is reused.
    """
    work = Path(out_dir) / "solvation"
    work.mkdir(parents=True, exist_ok=True)

    existing = sorted(work.glob("*_solv.prmtop"))
    if existing:
        prm = existing[0]
        rst = prm.with_suffix(".rst7")
        if rst.exists():
            return SolvationResult(prm, rst, work / "tleap_solvate.log", ok=True,
                                   warnings=["reused existing solvated system"])

    recipe = mol.files["leap_solvate.in"]
    recipe_text = recipe.read_text()
    warnings = _check_water_consistency(recipe_text, planned_water_model)
    warnings += mol.warnings

    # stage recipe + referenced inputs into the working dir
    shutil.copy2(recipe, work / "leap_solvate.in")
    for name in _MAYBE_INPUTS:
        src = mol.mol_dir / name
        if src.exists():
            shutil.copy2(src, work / name)

    log = work / "tleap_solvate.log"
    proc = subprocess.run(["tleap", "-f", "leap_solvate.in"], cwd=str(work),
                          capture_output=True, text=True, check=False)
    log.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)

    prms = sorted(work.glob("*_solv.prmtop"))
    if not prms:
        return SolvationResult(work / "none", work / "none", log, ok=False,
                               warnings=warnings + ["tleap produced no *_solv.prmtop"])
    prm = prms[0]
    rst = prm.with_suffix(".rst7")
    ok = prm.exists() and rst.exists() and prm.stat().st_size > 0
    return SolvationResult(prm, rst, log, ok=ok, warnings=warnings)
