#!/usr/bin/env python3
"""Generate the Colab MD notebook (notebooks/stage2b_md_colab.ipynb).

The notebook is a thin driver over `ensemble_qsar.md`: the validated pipeline
functions live in the package (and are unit-/dry-run-tested), while the notebook
handles the Colab-specific concerns (condacolab install, Drive, per-step cells,
batch). Regenerate with:  python scripts/build_md_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "stage2b_md_colab.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}


CELLS = [
    md("""# Stage 2b — MD simulation + ensemble analysis (Colab / OpenMM)

Takes the **prep hand-off** (`ligand.mol2`, `ligand.frcmod`, `leap_solvate.in`,
`manifest.json`, …) produced by Stage 2a and runs, per molecule:

**solvate → minimize → equilibrate → production → analysis**,

yielding a **conformational ensemble** and permeability-relevant descriptors
(**3D-PSA**, RMSD, PCA, clusters) for each molecule.

**Hardware:** set the runtime to **GPU** (Runtime ▸ Change runtime type ▸ T4 GPU).
Every step is **resume-safe** — if Colab disconnects, re-run the cell and it
continues from the last checkpoint (store outputs on Drive so they survive).

**Rough timing (T4, HMR 4 fs):** a small solvated ligand samples ~50–150 ns/day,
so a **1 ns test ≈ 10–30 min**; a **50 ns** production is multi-hour and may span
sessions. Large systems (e.g. the 956 Da macrocycle SDZ-RAD, ~big box) are
several-fold slower — start with the 1 ns test before scaling up."""),

    md("## 1 · Config — all tunables in one place\nEdit here; nothing below is hard-coded. Values the prep stage already decided (water model, seed) are merged from each molecule's `manifest.json`."),
    code("""
CONFIG = dict(
    # --- integrator / thermodynamics (HMR 4 fs) ---
    temperature_K=300.0, pressure_bar=1.0,
    timestep_fs=4.0, hmr=True, hydrogen_mass_amu=3.5, friction_ps=1.0,
    nonbonded_cutoff_nm=1.0, constraints="HBonds",
    # --- equilibration ---
    nvt_ns=0.1, npt_ns=0.2, restraint_schedule=(10.0, 5.0, 2.0, 0.0),
    # --- production (TEST value; raise to 50.0 once the flow is verified) ---
    production_ns=1.0, save_interval_ps=10.0, checkpoint_interval_ps=100.0,
    # --- platform ---
    platform="CUDA", precision="mixed", seed=42,
    # --- analysis ---
    pca_components=10, n_clusters=5, cluster_on="pca",
    descriptors=("psa3d", "rmsd"),   # permeability core; add "rg" if wanted
    # --- io (set after Drive mount below) ---
    input_root="", output_root="", use_drive=True,
)
"""),

    md("## 2 · Setup — condacolab\n`condacolab.install()` **restarts the kernel** (expected). Run this cell alone, wait for the restart, then continue from the next cell. Skip it if `conda` is already present."),
    code("""
try:
    import condacolab; condacolab.check()
    print("condacolab already installed")
except Exception:
    !pip -q install condacolab
    import condacolab; condacolab.install()   # kernel restarts here
"""),

    md("## 3 · Setup — packages + code\nInstall the MD stack from conda-forge (AmberTools gives `tleap`; OpenMM is the engine) and get the `ensemble_qsar` package + the prep hand-off files. The repo is the simplest source of both."),
    code("""
import os, sys, subprocess

# condacolab sometimes pins a python version (e.g. 3.12) that does not match the
# python it actually installed (e.g. 3.11), which makes `mamba install` refuse to
# run ("Your pinning does not match what's currently installed"). Realign the pin
# to the RUNNING interpreter so the solver keeps python fixed and installs
# compatible builds.
pin = f"python {sys.version_info.major}.{sys.version_info.minor}.*"
with open("/usr/local/conda-meta/pinned", "w") as fh:
    fh.write(pin + "\\n")
print("aligned conda pin ->", pin)

# MD stack (AmberTools for tleap solvation, OpenMM engine, analysis libs).
# Unpinned versions so the solver picks builds matching Colab's python.
!mamba install -q -y -c conda-forge openmm ambertools mdtraj mdanalysis scikit-learn matplotlib

# Get the code + prep hand-off files. For a PRIVATE repo, add a Colab secret
# named GITHUB_TOKEN (key icon on the left) with repo read scope.
REPO = "errory230/CV_LifeSciences"
BRANCH = "claude/ensemble-qsar-molecules-y18x5n"
token = ""
try:
    from google.colab import userdata
    token = userdata.get("GITHUB_TOKEN")
except Exception:
    pass
auth = f"{token}@" if token else ""
if not os.path.isdir("/content/CV_LifeSciences"):
    subprocess.run(["git", "clone", "-b", BRANCH,
                    f"https://{auth}github.com/{REPO}.git",
                    "/content/CV_LifeSciences"], check=True)
sys.path.insert(0, "/content/CV_LifeSciences")

import openmm, mdtraj, sklearn
from ensemble_qsar.md.config import MDConfig
from ensemble_qsar.md import io, solvate, simulate, analyze, pipeline
print("openmm", openmm.__version__, "| mdtraj", mdtraj.__version__)
"""),

    md("## 4 · Google Drive (optional but recommended)\nStore outputs on Drive so a disconnect doesn't lose the trajectory/checkpoints. Prep inputs are read from the cloned repo by default; point `input_root` at Drive instead if your prep outputs live there."),
    code("""
if CONFIG["use_drive"]:
    from google.colab import drive
    drive.mount("/content/drive")
    base = "/content/drive/MyDrive/mds_qsar"
    os.makedirs(base, exist_ok=True)
    CONFIG["output_root"] = f"{base}/md"
else:
    CONFIG["output_root"] = "/content/md"

# default: read prep hand-off files committed in the repo
CONFIG["input_root"] = "/content/CV_LifeSciences/data/prep"
os.makedirs(CONFIG["output_root"], exist_ok=True)
print("input_root :", CONFIG["input_root"])
print("output_root:", CONFIG["output_root"])
"""),

    md("## 5 · Load one molecule\nStart with the simplest (`T1_smoke`, pyridine) to verify the flow end-to-end."),
    code("""
cfg = MDConfig(**CONFIG)
MOL_DIR = f"{CONFIG['input_root']}/T1_smoke"

mol = io.load_molecule(MOL_DIR)
cfg = cfg.merge_manifest(mol.manifest)
print("mol_id :", mol.mol_id, "| class:", mol.mol_class)
print("net_charge:", mol.manifest.get("net_charge"), "| water:", cfg.water_model, "| seed:", cfg.seed)
print("hand-off:", list(mol.files))
"""),

    md("## 6 · Step 1 — Solvation (tleap)\nRuns the prep `leap_solvate.in`, building a solvated + neutralized box with the exact prep charges/parameters."),
    code("""
from pathlib import Path
out_dir = Path(cfg.output_root) / mol.mol_id
out_dir.mkdir(parents=True, exist_ok=True)

sol = solvate.solvate(mol, out_dir, planned_water_model=mol.manifest.get("water_model_planned", ""))
print("ok:", sol.ok, "| warnings:", sol.warnings)
import parmed; p = parmed.load_file(str(sol.prmtop))
print("solvated system:", len(p.atoms), "atoms,", len(p.residues), "residues")
"""),

    md("## 7 · Step 2 — Minimization"),
    code("""
mini = simulate.minimize(sol.prmtop, sol.rst7, out_dir, cfg)
print("ok:", mini.ok, "| info:", mini.info, "| seconds:", mini.seconds)
"""),

    md("## 8 · Step 3 — Equilibration (NVT heat → NPT, restraint release)"),
    code("""
eq = simulate.equilibrate(sol.prmtop, mini.output, out_dir, cfg)
print("ok:", eq.ok, "| info:", eq.info, "| seconds:", eq.seconds)
"""),

    md("## 9 · Step 4 — Production (resume-safe)\nWrites `trajectory.dcd` + `statedata.csv`, checkpointing to Drive. **If Colab disconnects, just re-run this cell** — it reloads `production.chk` and continues appending."),
    code("""
prod = simulate.produce(sol.prmtop, eq.output, out_dir, cfg)
print("ok:", prod.ok, "| info:", prod.info, "| seconds:", prod.seconds)
"""),

    md("## 10 · Step 5 — Analysis → ensemble\nStrip solvent, align to the prep reference, compute 3D-PSA/RMSD, PCA, cluster, and save one representative structure per cluster."),
    code("""
ana = analyze.analyze(prod.output, sol.prmtop, mol.reference_structure, out_dir, cfg)
print("frames:", ana.n_frames, "| representatives:", [p.name for p in ana.representatives])
print("warnings:", ana.warnings)

import pandas as pd
from IPython.display import Image, display
display(pd.read_csv(ana.metrics_csv).head())
for png in ana.plots:
    display(Image(str(png)))
"""),

    md("## 11 · Run manifest\nFull provenance: prep manifest + the actual MD parameters and results used."),
    code("""
rm = io.write_run_manifest(out_dir, mol, cfg.as_dict(),
                           {"trajectory": str(prod.output), "n_frames": ana.n_frames})
print("wrote", rm)
"""),

    md("## 12 · One call, or batch over many molecules (fail-soft)\n`run_molecule` chains all steps (resume-safe); `run_batch` loops molecules, skipping failures and logging them so the batch never halts."),
    code("""
import glob
# single molecule, all steps in one call:
# pipeline.run_molecule(MOL_DIR, cfg)

mol_dirs = sorted(d for d in glob.glob(f"{CONFIG['input_root']}/*") if os.path.isdir(d))
print("found", len(mol_dirs), "molecules")
report = pipeline.run_batch(mol_dirs, cfg)
import json; print(json.dumps(report, indent=2))
"""),

    md("""## 13 · Scaling up

1. **Verify** with `production_ns=1.0` (this notebook's default) on `T1_smoke`.
2. Raise `CONFIG["production_ns"]` to **50.0** and re-run — checkpoint/resume
   carries a long run across Colab sessions.
3. Point `input_root` at a Stage-1 representative set prepped in Stage 2a (e.g.
   `.../data/prep/caco2`) and use the batch cell.

**Watch:** free-Colab GPU time/idle limits; large macrocycles (bigger boxes) are
slower — consider running them in their own session. All outputs land under
`output_root/<mol_id>/` with a `run_manifest.json` recording exactly what ran."""),
]


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": [], "gpuType": "T4"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
