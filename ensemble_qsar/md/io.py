"""Input loading + run-manifest output for the MD stage."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


# prep hand-off files we expect per molecule (small-molecule route).
_EXPECTED_SMALL = ["leap_solvate.in", "ligand.mol2", "ligand.frcmod",
                   "ligand_gas.prmtop", "ligand_gas.rst7", "ligand_ref.pdb"]
_EXPECTED_PEPTIDE = ["leap_solvate.in", "peptide_gas.prmtop", "peptide_capped.pdb"]


@dataclass
class MolInput:
    mol_id: str
    mol_dir: Path
    manifest: dict
    files: dict[str, Path] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def mol_class(self) -> str:
        return self.manifest.get("mol_class", "small_molecule")

    @property
    def reference_structure(self) -> Path | None:
        for name in ("ligand_ref.pdb", "peptide_capped.pdb"):
            p = self.mol_dir / name
            if p.exists():
                return p
        return None


def load_molecule(mol_dir: str | Path) -> MolInput:
    """Load a prep molecule directory: manifest + hand-off files + checks."""
    mol_dir = Path(mol_dir)
    manifest_path = mol_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no manifest.json in {mol_dir}")
    manifest = json.loads(manifest_path.read_text())
    mol_id = manifest.get("mol_id", mol_dir.name)

    expected = _EXPECTED_PEPTIDE if manifest.get("mol_class") == "peptide" else _EXPECTED_SMALL
    files, warnings = {}, []
    for name in expected:
        p = mol_dir / name
        if p.exists():
            files[name] = p
        else:
            warnings.append(f"missing hand-off file: {name}")
    if "leap_solvate.in" not in files:
        raise FileNotFoundError(f"{mol_id}: leap_solvate.in required for solvation")

    return MolInput(mol_id=mol_id, mol_dir=mol_dir, manifest=manifest,
                    files=files, warnings=warnings)


def write_run_manifest(out_dir: Path, mol: MolInput, cfg_dict: dict,
                       actuals: dict) -> Path:
    """Extend the prep manifest with the actual MD run parameters + results."""
    data = {
        "mol_id": mol.mol_id,
        "prep_manifest": mol.manifest,
        "md_config": cfg_dict,
        "actuals": actuals,   # steps run, wall-times, seeds, n_frames, versions...
    }
    path = Path(out_dir) / "run_manifest.json"
    path.write_text(json.dumps(data, indent=2, default=str))
    return path
