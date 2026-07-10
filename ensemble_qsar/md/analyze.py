"""Step 5: trajectory analysis -> conformational ensemble (CPU).

Strips solvent, aligns the solute to the prep reference, computes the
permeability-relevant per-frame descriptors (3D-PSA = polar SASA, plus RMSD),
reduces the aligned heavy-atom coordinates with PCA, clusters the PCA space, and
saves one representative structure per cluster. Outputs a per-frame metrics CSV,
an ensemble summary, cluster representative PDBs, and (if matplotlib is present)
PCA/time-series plots.

3D-PSA definition: solvent-accessible surface area of polar atoms (N, O, and H
bonded to N/O), from Shrake-Rupley, in A^2 — the ensemble analogue of 2D TPSA
that captures a flexible molecule's ability to shield polar groups.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import mdtraj as md
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

_WATER = {"HOH", "WAT", "SOL"}
_IONS = {"NA", "CL", "K", "NA+", "CL-", "K+", "Na+", "Cl-", "K+"}


def _solute_indices(top) -> np.ndarray:
    keep = []
    for atom in top.atoms:
        rn = atom.residue.name
        if rn in _WATER or rn in _IONS:
            continue
        keep.append(atom.index)
    return np.array(keep, dtype=int)


def _polar_atom_mask(top, indices) -> np.ndarray:
    """Polar atoms among `indices`: N, O, and H bonded to N/O."""
    polar = []
    for i in indices:
        a = top.atom(int(i))
        sym = a.element.symbol if a.element else ""
        if sym in ("N", "O"):
            polar.append(True)
        elif sym == "H":
            is_polar_h = any(
                (n.element and n.element.symbol in ("N", "O"))
                for n in _bonded_atoms(top, a)
            )
            polar.append(is_polar_h)
        else:
            polar.append(False)
    return np.array(polar, dtype=bool)


def _bonded_atoms(top, atom):
    for b in top.bonds:
        if b[0].index == atom.index:
            yield b[1]
        elif b[1].index == atom.index:
            yield b[0]


@dataclass
class AnalysisResult:
    metrics_csv: Path
    summary_json: Path
    representatives: list[Path] = field(default_factory=list)
    plots: list[Path] = field(default_factory=list)
    n_frames: int = 0
    ok: bool = True
    warnings: list[str] = field(default_factory=list)


def analyze(traj_path, prmtop_path, ref_pdb, out_dir: Path, cfg) -> AnalysisResult:
    out_dir = Path(out_dir)
    ana = out_dir / "analysis"
    ana.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    traj = md.load(str(traj_path), top=str(prmtop_path))
    solute = _solute_indices(traj.topology)
    traj = traj.atom_slice(solute)
    top = traj.topology
    heavy = top.select("element != H")

    # Make the solute whole across periodic boundaries before alignment, so a
    # molecule imaged across the box edge in a frame does not appear as a huge
    # RMSD/PCA jump. Harmless for already-whole frames.
    try:
        traj.image_molecules(inplace=True)
    except Exception as e:  # noqa: BLE001
        warnings.append(f"image_molecules skipped ({e})")

    # --- alignment to the prep reference (fallback: first frame) -----------
    ref = None
    if ref_pdb and Path(ref_pdb).exists():
        try:
            ref = md.load(str(ref_pdb))
            if ref.n_atoms != traj.n_atoms:
                warnings.append("reference atom count != solute; using frame 0")
                ref = None
        except Exception as e:  # noqa: BLE001
            warnings.append(f"could not load reference ({e}); using frame 0")
    if ref is None:
        ref = traj[0]
    traj.superpose(ref, atom_indices=heavy)

    n = traj.n_frames
    time_ps = np.arange(n) * cfg.save_interval_ps

    # --- per-frame descriptors ---------------------------------------------
    cols = {"frame": np.arange(n), "time_ps": time_ps}
    if "rmsd" in cfg.descriptors:
        cols["rmsd_A"] = md.rmsd(traj, ref, atom_indices=heavy) * 10.0
    if "psa3d" in cfg.descriptors:
        sasa = md.shrake_rupley(traj, mode="atom")  # (n_frames, n_atoms) nm^2
        polar = _polar_atom_mask(top, np.arange(traj.n_atoms))
        cols["psa3d_A2"] = sasa[:, polar].sum(axis=1) * 100.0  # nm^2 -> A^2
    if "rg" in cfg.descriptors:
        cols["rg_A"] = md.compute_rg(traj) * 10.0

    # --- PCA on aligned heavy-atom coordinates -----------------------------
    X = traj.xyz[:, heavy, :].reshape(n, -1)
    n_pc = min(cfg.pca_components, max(1, min(X.shape) - 1))
    pca = PCA(n_components=n_pc).fit(X)
    proj = pca.transform(X)
    cols["PC1"] = proj[:, 0]
    if n_pc > 1:
        cols["PC2"] = proj[:, 1]

    # --- clustering + representatives --------------------------------------
    space = proj if cfg.cluster_on == "pca" else X
    k = min(cfg.n_clusters, n)
    labels = np.zeros(n, dtype=int)
    reps: list[Path] = []
    if k >= 1 and n >= k:
        km = KMeans(n_clusters=k, random_state=int(cfg.seed), n_init=10).fit(space)
        labels = km.labels_
        for c in range(k):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                continue
            centroid = km.cluster_centers_[c]
            nearest = members[np.argmin(np.linalg.norm(space[members] - centroid, axis=1))]
            rep = ana / f"cluster{c:02d}_frame{nearest}.pdb"
            traj[nearest].save_pdb(str(rep))
            reps.append(rep)
    cols["cluster"] = labels

    # --- write metrics CSV --------------------------------------------------
    metrics = ana / "metrics.csv"
    _write_csv(metrics, cols)

    # --- ensemble summary ---------------------------------------------------
    summary = {
        "n_frames": int(n),
        "n_clusters": int(k),
        "cluster_populations": {int(c): int((labels == c).sum()) for c in range(k)},
        "pca_explained_variance": [float(v) for v in pca.explained_variance_ratio_],
        "descriptors": {},
        "representative_frames": [int(p.stem.split("frame")[1]) for p in reps],
    }
    for key in ("rmsd_A", "psa3d_A2", "rg_A"):
        if key in cols:
            v = np.asarray(cols[key], float)
            summary["descriptors"][key] = {
                "mean": float(v.mean()), "std": float(v.std()),
                "min": float(v.min()), "max": float(v.max())}
    summary_path = ana / "ensemble_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    plots = _make_plots(ana, cols, labels, k) if _HAS_MPL else []
    if not _HAS_MPL:
        warnings.append("matplotlib not available; plots skipped")

    return AnalysisResult(metrics, summary_path, reps, plots, n_frames=n,
                          ok=True, warnings=warnings)


def _write_csv(path: Path, cols: dict) -> None:
    keys = list(cols)
    rows = zip(*[np.asarray(cols[k]) for k in keys])
    with path.open("w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{x:.5g}" if isinstance(x, float) else str(x) for x in r) + "\n")


try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:  # noqa: BLE001
    _HAS_MPL = False


def _make_plots(ana: Path, cols: dict, labels, k) -> list[Path]:
    out = []
    if "PC1" in cols and "PC2" in cols:
        fig, ax = plt.subplots(figsize=(5, 4))
        sc = ax.scatter(cols["PC1"], cols["PC2"], c=labels, cmap="tab10", s=12)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("Conformational PCA")
        fig.colorbar(sc, label="cluster"); fig.tight_layout()
        p = ana / "pca_scatter.png"; fig.savefig(p, dpi=120); plt.close(fig); out.append(p)
    series = [c for c in ("rmsd_A", "psa3d_A2", "rg_A") if c in cols]
    if series:
        fig, axes = plt.subplots(len(series), 1, figsize=(6, 2.2 * len(series)), squeeze=False)
        for ax, c in zip(axes[:, 0], series):
            ax.plot(cols["time_ps"], cols[c], lw=0.8)
            ax.set_ylabel(c); ax.set_xlabel("time (ps)")
        fig.suptitle("Per-frame descriptors"); fig.tight_layout()
        p = ana / "timeseries.png"; fig.savefig(p, dpi=120); plt.close(fig); out.append(p)
    return out
