"""Offline tests for the Stage-5 CLI helpers (no MD run)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ensemble_qsar.cli import orchestrate  # noqa: E402


def test_mol_id_deterministic_from_smiles():
    a = orchestrate.make_mol_id("c1ccncc1", None)
    b = orchestrate.make_mol_id("c1ccncc1", None)
    assert a == b and a.startswith("mol_") and len(a) == 12  # "mol_" + 8 hex


def test_mol_id_prefers_name_and_sanitizes():
    assert orchestrate.make_mol_id("CCO", "my mol/2") == "my_mol_2"


def test_configs_override_and_n_frames():
    prep, md = orchestrate._configs(
        {"pH": 6.5, "production_ns": 10.0, "n_frames": 500, "platform": "OpenCL"},
        Path("/tmp/x"))
    assert prep.ph == 6.5
    assert md.production_ns == 10.0 and md.platform == "OpenCL"
    # 500 frames over 10 ns -> save every 20 ps
    assert abs(md.save_interval_ps - 20.0) < 1e-6


def test_run_one_rejects_invalid_smiles():
    raised = False
    try:
        orchestrate.run_one("not_a_smiles", outdir=Path("/tmp/_cli_x"))
    except ValueError:
        raised = True
    assert raised


if __name__ == "__main__":
    test_mol_id_deterministic_from_smiles()
    test_mol_id_prefers_name_and_sanitizes()
    test_configs_override_and_n_frames()
    test_run_one_rejects_invalid_smiles()
    print("all cli tests passed")
