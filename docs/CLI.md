# Stage 5 — `run.py` (SMILES → MD-ensemble report)

A single-user local tool that runs the full pipeline for a molecule and produces
a self-contained conformation-ensemble explorer. It is a **thin orchestrator**:
it calls the existing Stage 2/3 code in order and adds no new pipeline logic.

```
SMILES ─▶ prep (2a) ─▶ MD (2b: solvate→min→equil→prod→analysis) ─▶ report (3) ─▶ explore_<id>.html
         ensemble_qsar.cli.orchestrate.run_one / run_many
```

## Requirements

Run inside the `mdsprep` conda env (`bash scripts/setup_conda_env.sh`). MD uses a
local GPU (CUDA→OpenCL→CPU auto-select); prep and analysis are CPU. The batch
progress bar uses `tqdm` if installed (optional).

## Usage

```bash
python run.py "O=C(O)c1ccccc1" --name aspirin        # single molecule
python run.py "c1ccncc1"                              # name defaults to a hash slug
python run.py --input molecules.csv --outdir results # batch (smiles [+ name] columns)

# config overrides (defaults come from PrepConfig / MDConfig)
python run.py "CCO" --production-ns 50 --n-frames 500 --pH 7.4 \
    --n-clusters 5 --water-model tip3p --platform CUDA --seed 42
python run.py "CCO" --force                           # re-run all stages
```

`--n-frames` sets the trajectory save interval (`save = production / n_frames`).

## Output — one merged directory per molecule

```
results/<mol_id>/
  ligand.mol2 · ligand.frcmod · ligand_gas.* · manifest.json   # prep (2a)
  solvation/ · trajectory.dcd · *.chk · statedata.csv · run_manifest.json  # MD (2b)
  analysis/  (metrics.csv · ensemble_summary.json · cluster*.pdb · plots)  # (3)
  viz_data.json · feature_detail.json
  explore_<mol_id>.html          # ← the report (self-contained, double-click)
  cli_run_manifest.json          # stages, timings, status, full MD config
results/
  index.html                     # hub over all molecules (batch)
  summary.csv                    # one row/molecule: status, report, key metrics
  _errors/<mol_id>.txt           # traceback for any failed molecule
```

`<mol_id>` is `--name` (sanitized) or `mol_<sha1(canonical SMILES)[:8]>` — the
hash slug is deterministic, so the same SMILES resumes into the same directory.

## Resume & fail-soft

- Each stage skips completed work (prep via its manifest, MD via on-disk
  checkpoints, report if `explore_*.html` already exists), so re-running the same
  command **resumes** in seconds. `--force` wipes the molecule directory first.
- In a batch, a failing molecule is logged to `_errors/` and skipped; the run
  continues and reports a success/failure summary at the end.
- Peptides whose sequence cannot be parsed from an opaque name fail-soft (the
  documented prep limitation).
