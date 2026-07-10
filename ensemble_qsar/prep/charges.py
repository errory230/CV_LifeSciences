"""Step: AM1-BCC partial charges + GAFF2 atom types via antechamber (CPU).

antechamber runs a semi-empirical AM1 optimization through `sqm`, then applies
the BCC bond-charge corrections — the standard, CPU-tractable charge model for
GAFF small molecules. The output mol2 carries both the charges and GAFF2 atom
types, ready for parmchk2 + tleap. Intermediate files (sqm.in/sqm.out, etc.) are
left in place for debugging.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ._util import CmdResult, run_cmd


@dataclass
class ChargeResult:
    mol2_path: Path
    net_charge: int
    charge_method: str
    ok: bool
    cmd: CmdResult


def assign_am1bcc_charges(
    sdf_path: Path,
    *,
    out_dir: Path,
    net_charge: int,
    atom_type: str = "gaff2",
) -> ChargeResult:
    """Run antechamber (AM1-BCC, GAFF2) on a 3D SDF; return the charged mol2."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mol2_path = out_dir / "ligand.mol2"

    cmd = [
        "antechamber",
        "-i", str(sdf_path), "-fi", "sdf",
        "-o", str(mol2_path), "-fo", "mol2",
        "-c", "bcc",              # AM1-BCC
        "-at", atom_type,         # GAFF2 atom types
        "-nc", str(net_charge),   # net molecular charge (must be correct)
        "-pf", "n",               # keep intermediates for debugging
        "-s", "2",
    ]
    res = run_cmd(cmd, cwd=out_dir, log_path=out_dir / "antechamber.log")
    ok = res.returncode == 0 and mol2_path.exists() and mol2_path.stat().st_size > 0

    return ChargeResult(
        mol2_path=mol2_path,
        net_charge=net_charge,
        charge_method="AM1-BCC (antechamber/sqm)",
        ok=ok,
        cmd=res,
    )
