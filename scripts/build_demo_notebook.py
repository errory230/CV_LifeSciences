#!/usr/bin/env python3
"""Generate the one-shot demo Colab notebook (notebooks/demo_pipeline_colab.ipynb).

A self-contained demonstration: type a SMILES + a few knobs, run all, and the
notebook goes SMILES -> Stage 2a prep -> Stage 2b MD -> self-contained HTML
report, showing a live progress bar for the (long) production step and rendering
the report inline.

Design (same philosophy as build_md_notebook.py): the validated pipeline lives in
the `ensemble_qsar` package; the notebook is a thin Colab driver. Two deliberate
choices keep it "run-all" smooth:

  * AmberTools binaries come from a **micromamba** prefix (no kernel restart,
    unlike condacolab) and are called as subprocesses; only their binaries matter.
  * OpenMM + RDKit + analysis libs are pip-installed into the *kernel* python.

Regenerate with:  python scripts/build_demo_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "demo_pipeline_colab.ipynb"

REPO = "errory230/CV_LifeSciences"
BRANCH = "claude/ensemble-qsar-molecules-y18x5n"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(keepends=True)}


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}


CELLS = [
    md("""# MDS-Ensemble QSAR — one-shot demo (SMILES → MD ensemble → HTML report)

Type a **SMILES** below, then **Runtime ▸ Run all**. The notebook runs the full
pipeline for that one molecule:

**Stage 2a** (prep: conformer · protonation at pH · AM1-BCC charges · GAFF2/ff19SB)
→ **Stage 2b** (MD: solvate → minimize → equilibrate → **production** → analysis)
→ **self-contained HTML report** (conformational ensemble, live 2D animation,
3D-PSA / PCA / t-SNE / clusters), rendered inline at the bottom.

* **Set the runtime to GPU:** Runtime ▸ Change runtime type ▸ **T4 GPU**.
* The long step (production MD) shows a **live progress bar**.
* Start with the default **1 ns** to see it end-to-end in a few minutes; raise
  `PRODUCTION_NS` for a real run.
* No kernel restart — setup uses `micromamba` for the AmberTools binaries, so
  *Run all* flows straight through."""),

    md("## 1 · Inputs\nEdit the form on the right (these are ordinary variables — the `#@param` comments just render the form)."),
    code("""
#@title  ← set your molecule and run parameters { display-mode: "form" }
SMILES        = "O=C(O)Cc1ccccc1"  #@param {type:"string"}
NAME          = "demo_mol"          #@param {type:"string"}
PH            = 7.4                  #@param {type:"number"}
PRODUCTION_NS = 1.0                 #@param {type:"number"}
N_FRAMES      = 100                 #@param {type:"integer"}
N_CLUSTERS    = 5                   #@param {type:"integer"}
USE_DRIVE     = False               #@param {type:"boolean"}
FORCE         = False               #@param {type:"boolean"}
print("SMILES:", SMILES, "| name:", NAME, "| pH:", PH,
      "| production:", PRODUCTION_NS, "ns |", N_FRAMES, "frames")
"""),

    md("""## 2 · Setup (no restart)

Installs the **AmberTools** binaries (antechamber / sqm / parmchk2 / tleap) into a
`micromamba` prefix — the prep step calls them as subprocesses, so only the
binaries matter, not their python. **OpenMM + RDKit + analysis libraries** are
pip-installed into *this kernel*. First run takes a few minutes; re-runs are fast."""),
    code(f"""
import os, sys, subprocess, shutil

PREFIX = "/content/mdsenv"
MAMBA  = "/content/bin/micromamba"

# --- micromamba (static binary; no kernel restart, unlike condacolab) ---
if not os.path.exists(MAMBA):
    os.makedirs("/content/bin", exist_ok=True)
    subprocess.run(
        "curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest "
        "| tar -C /content -xj bin/micromamba", shell=True, check=True)

# --- AmberTools binaries into a prefix (antechamber/sqm/parmchk2/tleap) ---
if not os.path.exists(f"{{PREFIX}}/bin/antechamber"):
    print("installing AmberTools (a few minutes) …")
    subprocess.run([MAMBA, "create", "-y", "-p", PREFIX,
                    "-c", "conda-forge", "ambertools"], check=True)
os.environ["PATH"] = f"{{PREFIX}}/bin:" + os.environ["PATH"]
os.environ["LD_LIBRARY_PATH"] = f"{{PREFIX}}/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ.setdefault("AMBERHOME", PREFIX)
for exe in ("antechamber", "sqm", "parmchk2", "tleap"):
    print(f"  {{exe}}: {{shutil.which(exe) or 'MISSING'}}")

# --- python libs into THIS kernel via pip (openmm ships the CUDA platform) ---
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "openmm", "rdkit", "mdtraj", "scikit-learn", "matplotlib",
                "scipy", "parmed", "dimorphite-dl", "pandas", "tqdm"], check=True)

import openmm
plats = [openmm.Platform.getPlatform(i).getName()
         for i in range(openmm.Platform.getNumPlatforms())]
print("openmm", openmm.__version__, "| platforms:", plats)
if "CUDA" not in plats and "OpenCL" not in plats:
    print("⚠ no GPU platform — Runtime ▸ Change runtime type ▸ GPU, then re-run.")

# --- code (public repo; for a private one add a Colab secret GITHUB_TOKEN) ---
REPO, BRANCH = "{REPO}", "{BRANCH}"
token = ""
try:
    from google.colab import userdata
    token = userdata.get("GITHUB_TOKEN")
except Exception:
    pass
auth = f"{{token}}@" if token else ""
if not os.path.isdir("/content/CV_LifeSciences"):
    subprocess.run(["git", "clone", "-b", BRANCH,
                    f"https://{{auth}}github.com/{{REPO}}.git",
                    "/content/CV_LifeSciences"], check=True)
if "/content/CV_LifeSciences" not in sys.path:
    sys.path.insert(0, "/content/CV_LifeSciences")
print("setup complete — code under /content/CV_LifeSciences")
"""),

    md("## 3 · Output location\nBy default results are written to `/content` (lost when the session ends). Tick `USE_DRIVE` in the form to persist them to Google Drive instead."),
    code("""
import os
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    OUTPUT_ROOT = "/content/drive/MyDrive/mds_qsar/demo"
else:
    OUTPUT_ROOT = "/content/demo_results"
os.makedirs(OUTPUT_ROOT, exist_ok=True)
print("OUTPUT_ROOT:", OUTPUT_ROOT)
"""),

    md("""## 4 · Run — prep → MD → report (live progress)

Calls the validated one-shot orchestrator (`orchestrate.run_one`) in a background
thread and polls `production_progress.json` to drive the production bar. Stage
transitions (prep · MD · report) stream above the bar. Every stage is
resume-safe, so re-running continues from the last checkpoint (tick `FORCE` to
start clean)."""),
    code("""
import threading, time, json, hashlib
from pathlib import Path
from tqdm.auto import tqdm
from rdkit import Chem
from ensemble_qsar.cli import orchestrate

mol = Chem.MolFromSmiles(SMILES)
assert mol is not None, f"invalid SMILES: {SMILES!r}"
canonical = Chem.MolToSmiles(mol)
mol_id  = orchestrate.make_mol_id(canonical, NAME or None)
outdir  = Path(OUTPUT_ROOT).resolve(); outdir.mkdir(parents=True, exist_ok=True)
mol_dir = outdir / orchestrate._safe(mol_id)
prog_f  = mol_dir / "production_progress.json"
if FORCE and prog_f.exists():
    prog_f.unlink()   # drop stale progress so the bar starts at 0

overrides = {k: v for k, v in dict(
    pH=PH, production_ns=PRODUCTION_NS, n_frames=N_FRAMES,
    n_clusters=N_CLUSTERS, platform="CUDA").items() if v is not None}

def _log(m): tqdm.write(str(m))

holder = {}
def _worker():
    try:
        holder["res"] = orchestrate.run_one(
            canonical, name=NAME or None, outdir=outdir,
            overrides=overrides, force=FORCE, log=_log)
    except Exception as e:  # surfaced below
        holder["err"] = e

print(f"▶ {mol_id}  ({canonical})")
t = threading.Thread(target=_worker); t.start()

pbar, seen = tqdm(total=100, desc="MD production", unit="%"), 0
while t.is_alive():
    if prog_f.exists():
        try:
            d = json.loads(prog_f.read_text())
            pct = min(100, int(100 * d.get("steps_done", 0) / max(1, d.get("target", 1))))
            if pct > seen:
                pbar.update(pct - seen); seen = pct
        except Exception:
            pass
    time.sleep(2)
t.join()
if seen < 100:
    pbar.update(100 - seen)
pbar.close()

res = holder.get("res")
if res is None or getattr(res, "status", "") != "ok":
    err = holder.get("err") or getattr(res, "error", "unknown error")
    print("\\n✗ FAILED:", err)
    print("  (see", outdir / "_errors", "for the traceback)")
else:
    secs = {k: v.get("seconds") for k, v in res.stages.items()}
    print("\\n✓ done — stage seconds:", secs)
    print("  report:", res.report)
"""),

    md("## 5 · The report, inline\nThe explorer HTML is fully self-contained (no external files); it renders here in an iframe. Drag the time slider to watch the conformation move, and use the PCA / t-SNE panels to explore the ensemble."),
    code("""
import html as _html
from pathlib import Path
from IPython.display import HTML, display

assert res is not None and res.status == "ok", "run cell 4 successfully first"
doc = Path(res.report).read_text()
display(HTML(
    f'<iframe srcdoc="{_html.escape(doc)}" width="100%" height="720" '
    f'style="border:1px solid #ccc;border-radius:6px"></iframe>'))
print("saved:", res.report)
# To download the report to your computer, uncomment:
# from google.colab import files; files.download(res.report)
"""),

    md("""## 6 · Notes

* **Timing (T4):** a small solvated ligand samples ~50–150 ns/day, so **1 ns ≈
  a few minutes**; raise `PRODUCTION_NS` for a real run (checkpoint/resume carries
  long runs across sessions when `USE_DRIVE` is on).
* **Batch / CLI:** for many molecules or a scripted run, use the repo CLI instead
  — `python run.py --input molecules.csv --outdir results` (builds a hub
  `index.html` + `summary.csv` over all molecules).
* **Bigger / flexible molecules** (macrocycles, peptides) use larger boxes and run
  several-fold slower — give them their own session and Drive output."""),
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
