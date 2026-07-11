"""Data exporter: a Stage-2 output directory -> viz_data.json (one per molecule).

This is layer (A). It is deliberately independent of the HTML viewer: it emits a
plain JSON document (frames, clusters, summary, embedded structure images) that
the molecule-agnostic viewer in `build_html` renders. Descriptors already in
`analysis/metrics.csv` (3D-PSA, RMSD, PCA/t-SNE, cluster) are reused as-is; the
rest (Rg, SASA, intra-molecular H-bonds) are recomputed from the trajectory.
"""

from __future__ import annotations

import json
from pathlib import Path

import mdtraj as md
import numpy as np
import pandas as pd

from ..features import descriptors3d
from ..md.analyze import _solute_indices
from . import render3d

_METRICS = ["psa3d", "rg", "intra_hbond", "sasa", "rmsd"]
# metrics.csv column -> viz field
_RENAME = {"psa3d_A2": "psa3d", "rmsd_A": "rmsd", "PC1": "pc1", "PC2": "pc2",
           "tSNE1": "tsne1", "tSNE2": "tsne2"}


def _versions() -> dict:
    import rdkit
    return {"mdtraj": getattr(md, "__version__", "?"), "rdkit": rdkit.__version__}


def _stats(v: np.ndarray) -> dict:
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "std": float(v.std()),
            "min": float(v.min()), "max": float(v.max()),
            "q25": float(np.percentile(v, 25)), "q50": float(np.percentile(v, 50)),
            "q75": float(np.percentile(v, 75))}


def build_viz_data(md_dir: Path, *, equilibration_frame: int = 0,
                   render_angles=None) -> dict:
    md_dir = Path(md_dir)
    metrics = pd.read_csv(md_dir / "analysis" / "metrics.csv").rename(columns=_RENAME)

    # provenance: smiles / save interval from the run (or prep) manifest
    manifest = {}
    for name in ("run_manifest.json", "manifest.json"):
        p = md_dir / name
        if p.exists():
            manifest = json.loads(p.read_text()); break
    prep = manifest.get("prep_manifest", manifest)
    smiles = prep.get("smiles", "")
    mol_id = prep.get("mol_id", md_dir.name)
    save_ps = float(manifest.get("md_config", {}).get("save_interval_ps", 0)) or \
        float(np.median(np.diff(metrics["time_ps"])) if len(metrics) > 1 else 0)

    # recompute Rg / SASA / intra_hbond from the solute trajectory
    prmtop = next((md_dir / "solvation").glob("*_solv.prmtop"))
    traj = md.load(str(md_dir / "trajectory.dcd"), top=str(prmtop))
    traj = traj.atom_slice(_solute_indices(traj.topology))
    try:
        traj.image_molecules(inplace=True)
    except Exception:
        pass
    extra = descriptors3d.frame_descriptors(traj)
    for k, arr in extra.items():
        metrics[k] = arr[: len(metrics)]

    # frames array
    fields = ["frame", "time_ps", "tsne1", "tsne2", "pc1", "pc2", "cluster",
              "psa3d", "rg", "intra_hbond", "sasa", "rmsd"]
    have = [f for f in fields if f in metrics.columns]
    frames = metrics[have].to_dict(orient="records")

    # clusters: rep image + mean metrics
    eq = metrics[metrics["frame"] >= equilibration_frame]
    clusters = []
    for pdb in sorted((md_dir / "analysis").glob("cluster*_frame*.pdb")):
        cid = int(pdb.stem.split("_")[0].replace("cluster", ""))
        rep_frame = int(pdb.stem.split("frame")[1])
        members = metrics[metrics["cluster"] == cid]
        clusters.append({
            "id": cid,
            "population": int(len(members)),
            "rep_frame": rep_frame,
            "images": render3d.render_structure(pdb, angles=render_angles),
            "mean_metrics": {m: float(members[m].mean()) for m in _METRICS if m in members},
        })

    # summary (over post-equilibration frames) + population-weighted ensemble mean
    summary = {m: _stats(eq[m]) for m in _METRICS if m in eq.columns}
    pops = np.array([c["population"] for c in clusters], float)
    weighted = {}
    if pops.sum() > 0:
        for m in _METRICS:
            if all(m in c["mean_metrics"] for c in clusters) and clusters:
                weighted[m] = float(np.average([c["mean_metrics"][m] for c in clusters], weights=pops))
    summary["ensemble_weighted"] = weighted
    ens = md_dir / "analysis" / "ensemble_summary.json"
    if ens.exists():
        summary["pca_explained_variance"] = json.loads(ens.read_text()).get("pca_explained_variance", [])

    return {
        "meta": {"mol_id": mol_id, "smiles": smiles, "n_frames": int(len(metrics)),
                 "equilibration_frame": int(equilibration_frame),
                 "save_interval_ps": save_ps, "versions": _versions()},
        "frames": frames,
        "clusters": clusters,
        "summary": summary,
        "descriptor_definitions": {
            "psa3d": {"definition": "SASA of N,O and polar H", "unit": "A^2"},
            "rg": {"definition": "radius of gyration", "unit": "A"},
            "sasa": {"probe_A": 1.4, "unit": "A^2"},
            "intra_hbond": {"donor_acceptor_A": 3.5, "angle_deg": 120,
                            "definition": "intramolecular H-bond count"},
            "rmsd": {"definition": "RMSD to prep reference (heavy atoms)", "unit": "A"},
        },
    }


def write_viz_data(md_dir: Path, out_path: Path | None = None, **kw) -> Path:
    data = build_viz_data(md_dir, **kw)
    out_path = Path(out_path) if out_path else Path(md_dir) / "viz_data.json"
    out_path.write_text(json.dumps(data))
    return out_path
