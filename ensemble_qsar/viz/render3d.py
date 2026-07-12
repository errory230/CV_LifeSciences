"""Headless 3D rendering of a conformer to base64 PNG(s) (matplotlib, no display).

Cluster representative structures (solute-only PDBs from the analysis stage) are
rendered as small static images embedded directly into the self-contained HTML
viewer. matplotlib's 3D axes need no display or extra dependency and run on any
CPU (verified in the project environment and on Colab), so this is the default
renderer. Each structure can be rendered from a few viewing angles.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

# Force a headless backend BEFORE importing matplotlib. Colab/IPython sets
# MPLBACKEND to an inline backend that is invalid under a plain `python` process,
# which makes `import matplotlib` itself raise; overriding the env var avoids it.
os.environ["MPLBACKEND"] = "Agg"
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from rdkit import Chem  # noqa: E402

# CPK-ish element colours and relative marker sizes.
_COLOR = {"C": "#3a3a3a", "N": "#3050f0", "O": "#e01010", "H": "#d0d0d0",
          "S": "#e6c000", "P": "#ff8000", "F": "#30c020", "Cl": "#20b020",
          "Br": "#a52a2a", "I": "#940094"}
_SIZE = {"H": 26}
_DEFAULT_ANGLES = [(18, -60), (18, 60)]  # (elev, azim)


def _load_conformer(pdb_path: Path):
    """Return (xyz Nx3 array, element symbols, bond index pairs)."""
    mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=False, proximityBonding=True)
    if mol is None or mol.GetNumConformers() == 0:
        return _parse_pdb_fallback(pdb_path)
    conf = mol.GetConformer()
    xyz = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    sym = [a.GetSymbol() for a in mol.GetAtoms()]
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    return xyz, sym, bonds


def _parse_pdb_fallback(pdb_path: Path):
    """Parse coords/elements from ATOM records; infer bonds by distance."""
    xyz, sym = [], []
    for line in Path(pdb_path).read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            el = line[76:78].strip() or line[12:16].strip()[0]
            sym.append(el.capitalize())
    xyz = np.array(xyz)
    bonds = []
    for i in range(len(xyz)):
        for j in range(i + 1, len(xyz)):
            d = np.linalg.norm(xyz[i] - xyz[j])
            cut = 1.3 if "H" in (sym[i], sym[j]) else 1.8
            if d < cut:
                bonds.append((i, j))
    return xyz, sym, bonds


def _render_one(xyz, sym, bonds, elev, azim, dpi=90) -> str:
    fig = plt.figure(figsize=(2.6, 2.6))
    ax = fig.add_subplot(111, projection="3d")
    for (x, y, z), s in zip(xyz, sym):
        ax.scatter(x, y, z, s=_SIZE.get(s, 90), c=_COLOR.get(s, "#888888"),
                   depthshade=True, edgecolors="#222", linewidths=0.3)
    for i, j in bonds:
        ax.plot(*zip(xyz[i], xyz[j]), c="#777777", lw=1.1)
    # equal aspect so the 3D shape is not distorted
    c = xyz.mean(axis=0)
    r = max(float(np.abs(xyz - c).max()), 1.0)
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, transparent=True)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def render_structure(pdb_path: Path, angles=None) -> list[str]:
    """Render a conformer PDB from the given angles; return base64 PNG data URIs."""
    xyz, sym, bonds = _load_conformer(Path(pdb_path))
    return [_render_one(xyz, sym, bonds, e, a) for e, a in (angles or _DEFAULT_ANGLES)]
