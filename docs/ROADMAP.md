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

### Stage 2b — GPU MD (next, off this environment)
- [ ] Run `leap_solvate.in` → solvate (explicit water; optional co-solvent/ions).
- [ ] Equilibrate + production MD (OpenMM) on GPU.
- [ ] Extract conformers two ways the user specified: (a) **time-uniform** —
      one frame per fixed interval; (b) **stability-triggered** — sample once
      RMSD/energy has plateaued.

### Stage 2c — CPU analysis (return trip)
- [ ] Align trajectory (RMSD), PCA over Cartesian/torsion space, cluster to a
      compact weighted ensemble; persist ensembles + provenance.

Design note: MD is the expensive step, which is why stage 1 selects a small,
flexibility-balanced subset and stage 2a keeps solvation off the CPU box.

## Stage 3 — Ensemble descriptors & QSAR

- [ ] Per-conformer 3D descriptors (shape, PSA/3D-PSA, radius of gyration,
      solvent-accessible surface, WHIM/GETAWAY, pharmacophore features).
- [ ] Aggregate to the molecule: Boltzmann-weighted means, spread, and
      distribution features across the ensemble.
- [ ] Train QSAR models (endpoint per dataset) on **static** vs **ensemble**
      features with identical splits.

## Stage 4 — Proof of concept: does the ensemble help?

- [ ] Compare static vs ensemble QSAR with a scaffold-aware split.
- [ ] **Stratify the gain by `flex_class`** — the core claim is that the
      ensemble advantage concentrates in flexible / macrocyclic molecules.
- [ ] Report, with ablations (sampling rule, ensemble size, pH sensitivity).
