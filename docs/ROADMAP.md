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

## Stage 2 — Conformer ensembles via solvent MD (next)

- [ ] Parametrize each representative molecule (e.g. OpenFF / GAFF), assign
      protonation state at a **configurable pH** (Dimorphite-DL / pKa model).
- [ ] Solvate (explicit water; optional co-solvent / ions), equilibrate, run
      production MD (OpenMM).
- [ ] Extract conformers by two strategies the user specified:
      (a) **time-uniform** — one frame per fixed interval; and
      (b) **stability-triggered** — sample once RMSD/energy has plateaued.
- [ ] Cluster conformers (RMSD) to a compact, weighted representative ensemble.
- [ ] Persist ensembles + provenance (force field, solvent, pH, sampling rule).

Design note: MD is the expensive step, which is why stage 1 selects a small,
flexibility-balanced subset rather than the full dataset.

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
