# Roadmap — MDS-Ensemble QSAR

The project tests one hypothesis: **conformational-ensemble descriptors from
solvent MD beat single-conformer descriptors for flexible molecules, and the
gain scales with flexibility.** Work is staged so each stage produces a
committed, inspectable artefact.

## Stage 1 — Data extraction & representative-set selection ✅ (implemented)

- [x] Pin data-rich public benchmarks for the two target endpoints
      (Caco-2 permeability, aqueous solubility) to reachable, checksummed
      mirrors — extensible registry in `ensemble_qsar/data/sources.py`.
- [x] Structure curation: parent/neutral/canonical, dedup, auditable report.
- [x] Flexibility + physicochemical descriptors, with a `flex_class` label.
- [x] Flexibility-stratified, diversity-based representative set for MD.
- [x] Offline sanity tests.

**Deliverable:** `data/processed/<dataset>/{curated.csv, representative_set.csv,
summary.json}`.

### Open decisions carried into later stages
- **Scale of the solubility arm.** ESOL (~1.1k) is small and mostly rigid.
  AqSolDB (~10k) or a ChEMBL/TDC pull would strengthen the solubility side and
  add flexible molecules — blocked here only by network policy, easy to add a
  registry entry when a source is reachable.
- **Where flexibility actually matters.** The `caco2_wang` set already carries
  ~261 beyond-Ro5 and ~81 macrocyclic molecules — the natural test bed for the
  ensemble hypothesis. A dedicated macrocycle/peptide set (e.g. CycPeptMPDB)
  is a strong candidate for a stage-3 stress test.

## Stage 2 — Conformer ensembles via solvent MD

Split by hardware: prep is CPU-only and completes here; dynamics needs a GPU.

### Stage 2a — CPU prep (SMILES → GPU-ready parameters) ✅ implemented
See [`PREP.md`](PREP.md). `ensemble_qsar/prep/` + `scripts/02_prep_molecule.py`,
`scripts/03_prep_batch.py`.
- [x] conda env with AmberTools (`environment.yml`, `setup_conda_env.sh`).
- [x] class routing (peptide vs small molecule) with review flags + evidence.
- [x] protonation at configurable pH (Dimorphite-DL; peptide library states).
- [x] ETKDG conformer + MMFF minimization.
- [x] AM1-BCC charges + GAFF2 (`mol2`+`frcmod`) / ff19SB peptide build.
- [x] dry, solvation-ready handoff + `leap_solvate.in` + provenance manifest.
- [x] fail-soft batch driver; validated on the 6-molecule validation set.

### Stage 2b — GPU MD + ensemble analysis ✅ implemented (Colab)
`ensemble_qsar/md/` + `notebooks/stage2b_md_colab.ipynb` (condacolab + OpenMM).
Logic validated end-to-end on a CPU OpenMM platform (T1 dry-run).
- [x] Run `leap_solvate.in` → solvate (explicit water + neutralizing ions) with
      the exact prep charges/parameters.
- [x] Minimize → equilibrate (NVT heat → NPT, restraint release) → production MD.
- [x] Time-uniform conformer capture (frame per fixed interval); all params from
      a top config block / prep manifest, actual values recorded.
- [x] Checkpoint/resume-safe (survives Colab disconnects); fail-soft batch.
- [x] Analysis (return trip): strip solvent, align to prep reference, per-frame
      **3D-PSA** + RMSD, PCA, KMeans clustering → representative conformers +
      metrics CSV + plots + `run_manifest.json`.
- [ ] Optional: stability-triggered sampling (sample once RMSD/energy plateaus)
      as an alternative to time-uniform — future refinement.

Design note: MD is the expensive step, which is why stage 1 selects a small,
flexibility-balanced subset and stage 2a keeps solvation off the CPU box.

## Stage 3 — Ensemble descriptors + explorer 🟡 (descriptors done; QSAR next)

`ensemble_qsar/features/`, `ensemble_qsar/viz/`, `scripts/20_export_viz.py`,
`scripts/21_build_all.py`. Verified end-to-end on a Stage-2 output.
- [x] Per-frame 3D descriptors: 3D-PSA, Rg, SASA, intramolecular H-bonds (with
      recorded criteria), plus PCA/t-SNE/RMSD from Stage 2.
- [x] Aggregate to a per-molecule feature row (distribution stats + population-
      weighted ensemble mean) → `features.csv`; Stage-1 label join by `mol_id`.
- [x] **Self-contained HTML explorer** per molecule (data exporter →
      `viz_data.json` → molecule-agnostic viewer): t-SNE/PCA scatter (click a
      conformer), time slider + play, descriptor time series, cluster
      representative 3D images (matplotlib, base64-embedded). Single file, no
      server/CDN. `index.html` links all molecules.
- [ ] Per-conformer WHIM/GETAWAY/pharmacophore features (optional richer set).
- [ ] Train QSAR models (endpoint per dataset) on **static** vs **ensemble**
      features with identical splits.

## Stage 5 — One-command CLI tool ✅ implemented

`run.py` + `ensemble_qsar/cli/orchestrate.py`. Thin orchestrator over Stages 2–3:
SMILES → prep → MD → analysis → self-contained `explore_<mol_id>.html`, in a
merged `results/<mol_id>/` dir. Deterministic hash-slug mol ids, resume-on-rerun
(stage skip), `--force`, fail-soft batch with `summary.csv` + hub. Validated
end-to-end on CPU. See [`CLI.md`](CLI.md).

## Stage 4 — Proof of concept: does the ensemble help?

- [ ] Compare static vs ensemble QSAR with a scaffold-aware split.
- [ ] **Stratify the gain by `flex_class`** — the core claim is that the
      ensemble advantage concentrates in flexible / macrocyclic molecules.
- [ ] Report, with ablations (sampling rule, ensemble size, pH sensitivity).
