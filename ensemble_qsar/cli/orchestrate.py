"""Stage 5 orchestrator: SMILES -> MD-ensemble HTML report.

A thin conductor over the existing stages — it does NOT reimplement any pipeline
logic. Per molecule it calls, in order:

    prep.pipeline.prep_molecule    (Stage 2a: SMILES -> GAFF2/ff19SB parameters)
    md.pipeline.run_molecule       (Stage 2b: solvate -> ... -> analysis)
    viz.export / build_html        (Stage 3: viz_data.json + explore_<id>.html)
    features.aggregate             (Stage 3: feature_detail.json)

Everything for one molecule lands in a single merged directory
`outdir/<mol_id>/`. Each underlying stage already skips completed work (prep via
its manifest, MD via on-disk checkpoints), so re-running resumes; `force` wipes
the molecule dir first. Batch runs are fail-soft with a summary + hub at the end.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from rdkit import Chem

from ..features import aggregate
from ..md.config import MDConfig
from ..md.pipeline import run_molecule
from ..prep.pipeline import PrepConfig, prep_molecule
from ..viz import build_hub, build_html, export


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def make_mol_id(canonical_smiles: str, name: str | None) -> str:
    if name:
        return _safe(name)
    return "mol_" + hashlib.sha1(canonical_smiles.encode()).hexdigest()[:8]


def _configs(overrides: dict, outdir: Path) -> tuple[PrepConfig, MDConfig]:
    o = overrides
    prep = PrepConfig(
        ph=o.get("pH", 7.4),
        random_seed=o.get("seed", 42),
        water_model=o.get("water_model", "tip3p"),
    )
    md_kw = dict(
        production_ns=o.get("production_ns", 20.0),
        water_model=o.get("water_model", "tip3p"),
        platform=o.get("platform", "CUDA"),
        seed=o.get("seed", 42),
        n_clusters=o.get("n_clusters", 5),
        output_root=str(outdir),
    )
    if o.get("save_interval_ps"):
        md_kw["save_interval_ps"] = o["save_interval_ps"]
    elif o.get("n_frames"):
        md_kw["save_interval_ps"] = max(
            0.1, md_kw["production_ns"] * 1000.0 / o["n_frames"])
    return prep, MDConfig(**md_kw)


@dataclass
class MolResult:
    mol_id: str
    smiles: str
    name: str | None = None
    status: str = ""
    report: str | None = None
    error: str | None = None
    stages: dict = field(default_factory=dict)
    viz: dict | None = None


def run_one(smiles: str, *, name: str | None = None, outdir: Path,
            overrides: dict | None = None, force: bool = False, log=print) -> MolResult:
    overrides = overrides or {}
    outdir = Path(outdir).resolve()   # absolute: external tools run with cwd set
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    canonical = Chem.MolToSmiles(mol)
    mol_id = make_mol_id(canonical, name)
    mol_dir = outdir / _safe(mol_id)
    res = MolResult(mol_id=mol_id, smiles=canonical, name=name)

    if force and mol_dir.exists():
        shutil.rmtree(mol_dir)
    prep_cfg, md_cfg = _configs(overrides, outdir)

    def _stage(step: str, fn):
        t0 = time.perf_counter()
        out = fn()
        res.stages[step] = {"status": "ok", "seconds": round(time.perf_counter() - t0, 1)}
        return out

    try:
        log(f"[{mol_id}] 1/3 prep …")
        m = _stage("prep", lambda: prep_molecule(mol_id, canonical, out_root=outdir, config=prep_cfg))
        res.stages["prep"]["mol_class"] = m.mol_class

        log(f"[{mol_id}] 2/3 MD ({md_cfg.production_ns} ns, {md_cfg.platform}) …")
        _stage("md", lambda: run_molecule(mol_dir, md_cfg))

        if not force and (mol_dir / f"explore_{mol_id}.html").exists() and (mol_dir / "viz_data.json").exists():
            log(f"[{mol_id}] 3/3 report (cached)")
            res.viz = json.loads((mol_dir / "viz_data.json").read_text())
        else:
            log(f"[{mol_id}] 3/3 analysis + report …")
            res.viz = _stage("analysis", lambda: _report(mol_dir, mol_id))

        res.status = "ok"
        res.report = str(mol_dir / f"explore_{mol_id}.html")
        _write_cli_manifest(mol_dir, res, md_cfg)
        log(f"[{mol_id}] ✓ done → {res.report}")
    except Exception as e:  # noqa: BLE001 fail-soft
        res.status = "failed"
        res.error = f"{type(e).__name__}: {e}"
        errdir = outdir / "_errors"; errdir.mkdir(parents=True, exist_ok=True)
        (errdir / f"{_safe(mol_id)}.txt").write_text(traceback.format_exc())
        log(f"[{mol_id}] ✗ FAILED at "
            f"{next((s for s in ('prep','md','analysis') if s not in res.stages), '?')}: {res.error}")
    return res


def _report(mol_dir: Path, mol_id: str) -> dict:
    viz = export.build_viz_data(mol_dir)
    (mol_dir / "viz_data.json").write_text(json.dumps(viz))
    build_html.build_html(viz, mol_dir / f"explore_{mol_id}.html")
    aggregate.write_feature_detail(viz, mol_dir / "feature_detail.json")
    return viz


def _write_cli_manifest(mol_dir: Path, res: MolResult, md_cfg: MDConfig) -> None:
    (mol_dir / "cli_run_manifest.json").write_text(json.dumps({
        "mol_id": res.mol_id, "name": res.name, "smiles": res.smiles,
        "status": res.status, "stages": res.stages,
        "md_config": md_cfg.as_dict(),
    }, indent=2))


def run_many(rows: list[dict], *, outdir: Path, overrides: dict | None = None,
             force: bool = False, log=print, progress=None) -> list[MolResult]:
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    results: list[MolResult] = []
    it = enumerate(rows, 1)
    if progress is not None:
        it = progress(it, total=len(rows))
    for i, row in it:
        smi = row.get("smiles") or row.get("SMILES")
        nm = row.get("name") or row.get("Name")
        if not smi:
            continue
        log(f"=== [{i}/{len(rows)}] {nm or smi[:24]} ===")
        try:
            results.append(run_one(smi, name=nm, outdir=outdir, overrides=overrides, force=force, log=log))
        except Exception as e:  # noqa: BLE001 (invalid SMILES etc.)
            results.append(MolResult(mol_id="?", smiles=smi, name=nm, status="failed",
                                     error=f"{type(e).__name__}: {e}"))
            log(f"  ✗ {e}")

    _write_summary(results, outdir)
    build_hub.build_hub(outdir)   # regenerate the Stage-3 hub over all molecules
    n_ok = sum(r.status == "ok" for r in results)
    log(f"\n== {n_ok}/{len(results)} ok · summary {outdir/'summary.csv'} · hub {outdir/'index.html'} ==")
    return results


def _write_summary(results: list[MolResult], outdir: Path) -> None:
    import csv
    rows = []
    for r in results:
        row = {"mol_id": r.mol_id, "name": r.name or "", "smiles": r.smiles,
               "status": r.status, "report": r.report or "", "error": r.error or ""}
        if r.viz:
            feat = aggregate.molecule_features(r.viz)
            for k in ("psa3d_mean", "psa3d_std", "rg_mean", "intra_hbond_mean", "rmsd_mean"):
                row[k] = feat.get(k)
        rows.append(row)
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with (outdir / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
