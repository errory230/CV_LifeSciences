"""Offline tests for the MD stage: config math + input loading (no MD run)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]


def test_config_step_math():
    from ensemble_qsar.md.config import MDConfig

    cfg = MDConfig(timestep_fs=4.0, save_interval_ps=10.0, checkpoint_interval_ps=100.0)
    assert cfg.n_steps(1.0) == 250_000          # 1 ns / 4 fs
    assert cfg.save_interval_steps() == 2_500   # 10 ps / 4 fs
    assert cfg.checkpoint_interval_steps() == 25_000


def test_config_merges_manifest():
    from ensemble_qsar.md.config import MDConfig

    cfg = MDConfig(water_model="tip3p", seed=1)
    merged = cfg.merge_manifest({"water_model_planned": "tip4pew", "random_seed": 7})
    assert merged.water_model == "tip4pew" and merged.seed == 7
    assert isinstance(merged.restraint_schedule, tuple)  # survives round-trip


def test_load_molecule_reads_prep_output():
    from ensemble_qsar.md.io import load_molecule

    d = ROOT / "data" / "prep" / "T1_smoke"
    if not (d / "manifest.json").exists():
        return  # prep output not present in this checkout; skip
    mol = load_molecule(d)
    assert mol.mol_id == "T1_smoke"
    assert "leap_solvate.in" in mol.files
    assert mol.reference_structure is not None


if __name__ == "__main__":
    test_config_step_math()
    test_config_merges_manifest()
    test_load_molecule_reads_prep_output()
    print("all md offline tests passed")
