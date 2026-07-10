"""MD stage orchestrator: run one molecule (resume-aware) or a batch.

`run_molecule` chains solvate → minimize → equilibrate → produce → analyze,
each writing under `output_root/<mol_id>/`. Every step is skip/resume-safe, so
re-invoking after a Colab disconnect continues rather than restarts. `run_batch`
loops molecules fail-soft: one failure is logged and skipped, the batch goes on.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path

from . import analyze, io, simulate, solvate
from .config import MDConfig


def run_molecule(mol_dir, cfg: MDConfig) -> dict:
    mol = io.load_molecule(mol_dir)
    cfg = cfg.merge_manifest(mol.manifest)
    out_dir = Path(cfg.output_root) / _safe(mol.mol_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    actuals: dict = {"steps": [], "warnings": list(mol.warnings)}

    def stage(name, fn):
        t0 = time.perf_counter()
        res = fn()
        actuals["steps"].append({
            "name": name, "ok": getattr(res, "ok", True),
            "seconds": round(time.perf_counter() - t0, 1),
            "info": getattr(res, "info", None) or {},
            "warnings": getattr(res, "warnings", []) or []})
        return res

    sol = stage("solvate", lambda: solvate.solvate(
        mol, out_dir, planned_water_model=mol.manifest.get("water_model_planned", "")))
    if not sol.ok:
        raise RuntimeError(f"{mol.mol_id}: solvation failed ({sol.warnings})")

    mini = stage("minimize", lambda: simulate.minimize(sol.prmtop, sol.rst7, out_dir, cfg))
    eq = stage("equilibrate", lambda: simulate.equilibrate(sol.prmtop, mini.output, out_dir, cfg))
    prod = stage("produce", lambda: simulate.produce(sol.prmtop, eq.output, out_dir, cfg))
    if not prod.ok:
        raise RuntimeError(f"{mol.mol_id}: production produced no trajectory")

    ana = stage("analyze", lambda: analyze.analyze(
        prod.output, sol.prmtop, mol.reference_structure, out_dir, cfg))

    actuals["topology"] = str(sol.prmtop)
    actuals["trajectory"] = str(prod.output)
    actuals["n_frames"] = ana.n_frames
    actuals["representatives"] = [str(p) for p in ana.representatives]
    io.write_run_manifest(out_dir, mol, cfg.as_dict(), actuals)
    return {"mol_id": mol.mol_id, "status": "ok", "out_dir": str(out_dir),
            "n_frames": ana.n_frames}


def run_batch(mol_dirs, cfg: MDConfig) -> list[dict]:
    report = []
    for i, d in enumerate(mol_dirs, 1):
        name = Path(d).name
        print(f"[{i}/{len(mol_dirs)}] {name} ... ", end="", flush=True)
        try:
            r = run_molecule(d, cfg)
            print(f"ok ({r['n_frames']} frames)")
        except Exception as e:  # noqa: BLE001 fail-soft
            r = {"mol_id": name, "status": "failed", "reason": f"{type(e).__name__}: {e}"}
            errdir = Path(cfg.output_root) / "_errors"
            errdir.mkdir(parents=True, exist_ok=True)
            (errdir / f"{i:03d}_{_safe(name)}.txt").write_text(traceback.format_exc())
            print(f"FAILED ({type(e).__name__})")
        report.append(r)
    return report


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)
