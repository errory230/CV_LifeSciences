"""Step: GAFF2 parameterization + GPU-handoff files (parmchk2 + tleap, CPU).

From the AM1-BCC-charged mol2 we generate the missing GAFF2 parameters
(`parmchk2` -> frcmod) and validate completeness by building a *gas-phase*
prmtop/rst7 with tleap. Solvation is intentionally NOT done here — it happens in
the GPU environment — so we also emit a templated `leap_solvate.in` recipe that
the GPU side runs to build the solvated system. The dry mol2 + frcmod + recipe
is the portable, engine-agnostic handoff.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ._util import run_cmd

# tleap recipe that only validates the parameters (gas phase, no solvent).
_LEAP_GAS = """source leaprc.gaff2
LIG = loadmol2 {mol2}
loadamberparams {frcmod}
check LIG
saveamberparm LIG {prmtop} {rst7}
savepdb LIG {refpdb}
quit
"""

# Templated recipe for the GPU side to solvate + neutralize. `addions ... 0`
# adds just enough counter-ions to neutralize the net charge.
_LEAP_SOLVATE = """# Run on the GPU host to build the solvated system.
source leaprc.gaff2
source leaprc.water.{water_leaprc}
LIG = loadmol2 {mol2}
loadamberparams {frcmod}
solvateBox LIG {water_box} {padding}
addions LIG {counterion} 0
saveamberparm LIG ligand_solv.prmtop ligand_solv.rst7
savepdb LIG ligand_solv.pdb
quit
"""

_WATER_LEAPRC = {"tip3p": ("tip3p", "TIP3PBOX"), "tip4pew": ("tip4pew", "TIP4PEWBOX")}


@dataclass
class ParamResult:
    frcmod_path: Path
    gas_prmtop: Path
    gas_rst7: Path
    ref_pdb: Path
    solvate_recipe: Path
    ok: bool
    messages: list[str] = field(default_factory=list)


def parameterize_gaff2(
    mol2_path: Path,
    *,
    out_dir: Path,
    net_charge: int,
    water_model: str = "tip3p",
    box_padding: float = 12.0,
) -> ParamResult:
    """parmchk2 -> frcmod, tleap gas-phase validation, and solvate recipe."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frcmod = out_dir / "ligand.frcmod"
    gas_prmtop = out_dir / "ligand_gas.prmtop"
    gas_rst7 = out_dir / "ligand_gas.rst7"
    ref_pdb = out_dir / "ligand_ref.pdb"
    messages: list[str] = []

    # 1) missing-parameter check -> frcmod
    pk = run_cmd(
        ["parmchk2", "-i", str(mol2_path), "-f", "mol2", "-o", str(frcmod), "-s", "gaff2"],
        cwd=out_dir,
        log_path=out_dir / "parmchk2.log",
    )
    if pk.returncode != 0 or not frcmod.exists():
        return ParamResult(frcmod, gas_prmtop, gas_rst7, ref_pdb,
                           out_dir / "leap_solvate.in", ok=False,
                           messages=["parmchk2 failed"])
    # frcmod entries flagged ATTN indicate guessed parameters worth reviewing.
    attn = [ln for ln in frcmod.read_text().splitlines() if "ATTN" in ln]
    if attn:
        messages.append(f"parmchk2: {len(attn)} parameter(s) flagged ATTN (guessed)")

    # 2) gas-phase build to prove the parameter set is complete
    leap_gas = out_dir / "leap_gas.in"
    leap_gas.write_text(
        _LEAP_GAS.format(mol2=mol2_path.name, frcmod=frcmod.name,
                         prmtop=gas_prmtop.name, rst7=gas_rst7.name, refpdb=ref_pdb.name)
    )
    tl = run_cmd(["tleap", "-f", leap_gas.name], cwd=out_dir, log_path=out_dir / "tleap_gas.log")
    ok = gas_prmtop.exists() and gas_prmtop.stat().st_size > 0 and gas_rst7.exists()
    if not ok:
        messages.append("tleap gas-phase build failed")

    # 3) solvation recipe for the GPU host (not executed here)
    leaprc, boxword = _WATER_LEAPRC.get(water_model, _WATER_LEAPRC["tip3p"])
    counterion = "Na+" if net_charge < 0 else "Cl-"
    solvate = out_dir / "leap_solvate.in"
    solvate.write_text(
        _LEAP_SOLVATE.format(
            water_leaprc=leaprc, mol2=mol2_path.name, frcmod=frcmod.name,
            water_box=boxword, padding=box_padding, counterion=counterion,
        )
    )

    return ParamResult(frcmod, gas_prmtop, gas_rst7, ref_pdb, solvate, ok=ok, messages=messages)
