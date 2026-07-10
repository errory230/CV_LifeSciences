# Stage 2a — CPU MD Prep (SMILES → GPU-ready parameters)

The molecular-dynamics work is split so the expensive part needs a GPU and the
rest does not:

```
  CPU (this stage)                    GPU (later)              CPU (later)
  ────────────────                    ───────────              ──────────
  SMILES → 3D → protonate →           solvate → equilibrate →  align → PCA →
  charges → parameterize      ──▶     production MD      ──▶   cluster →
  = dry, parameterized input          = trajectory             ensemble
```

This stage is **complete on CPU alone**: it turns each Stage-1 molecule into a
parameterized, solvation-ready input plus a `leap_solvate.in` recipe the GPU host
runs. No GPU, no solvation, no dynamics here.

## Environment

External parameterization tools (AmberTools) ship on conda-forge, so the stage
runs in a conda env, not pip:

```bash
bash scripts/setup_conda_env.sh          # Miniconda + `mdsprep` env
conda activate mdsprep
```

`environment.yml` pins it: `ambertools=22` (antechamber/sqm/parmchk2/tleap),
`openmm`, `rdkit`, `parmed`, `pdb2pqr`/`propka`, and `dimorphite-dl` (pip).

## Molecule-class routing (`prep/classify.py`)

Parameterization branches on structure, decided by transparent RDKit/SMARTS
rules, with every ambiguous case flagged (`needs_review`) and its evidence logged:

| Signal | SMARTS / rule | Used for |
|---|---|---|
| backbone residue | `[NX3][CX4;H1,H2][CX3](=O)` (covers Gly) | ≥3 ⇒ **peptide** route |
| amide bond | `[NX3][CX3](=O)` | evidence |
| macrocycle | largest ring ≥ 12 | review (ring topology) |
| ester | `[CX3](=O)[OX2H0][#6]` | depsipeptide review |
| N-methyl amide | `[CX3](=O)[NX3]([CH3])` | non-standard residue review |
| carbohydrate-like | ≥2 ring-O `[#6;R][OX2;R][#6;R]` + ≥3 `[OX2H]` | GLYCAM-recommended review |

Non-peptides (including macrocyclic natural products such as everolimus) take the
small-molecule route; only genuine polypeptides take the peptide route.

## Pipeline steps (`prep/pipeline.py`, idempotent & resumable)

Each step's status is recorded in the molecule's `manifest.json`; a completed
step is skipped on re-run. External tools run in `intermediates/` (debug files
preserved) and the handoff files are copied up.

**Small-molecule route (GAFF2 + AM1-BCC):**
1. **protonate** — dominant microstate at pH (default 7.4) via Dimorphite-DL,
   queried at a single pH point; implausible variants (e.g. deprotonated amides)
   are rejected.
2. **conformer** — ETKDGv3 multi-embed + MMFF94s minimization → lowest-energy 3D.
3. **charges** — AM1-BCC via `antechamber`/`sqm`, GAFF2 atom types → `ligand.mol2`.
4. **parameterize** — `parmchk2` → `ligand.frcmod`; `tleap` gas-phase build
   (validates completeness) → `ligand_gas.prmtop/rst7`; emit `leap_solvate.in`.

> Note: protonation runs before 3D embedding (it changes the atom set / net
> charge the later steps depend on), a deliberate reordering of the numbered spec.

**Peptide route (ff19SB library charges):** parse the sequence from the peptide
name → `tleap` sequence build with ff19SB (standard residue names already encode
pH-7.4 states: LYS+/ARG+/ASP−/GLU−/HIS=HIE); macrocyclic peptides get an explicit
head-to-tail `bond`. The built MW is cross-checked against the SMILES to catch
name/structure mismatch. Automatic SMILES→sequence perception for unnamed
peptides is future work; such inputs are flagged for review and skipped.

## Handoff files (per `data/prep/<mol_id>/`)

| Small molecule | Peptide | Both |
|---|---|---|
| `ligand.mol2` (AM1-BCC + GAFF2) | `peptide_gas.prmtop/rst7` | `leap_solvate.in` (GPU recipe) |
| `ligand.frcmod` | `peptide_capped.pdb` | `manifest.json` (provenance + ledger) |
| `ligand_gas.prmtop/rst7`, `ligand_ref.pdb`, `ligand.sdf` | | `intermediates/` (debug) |

`manifest.json` records: mol class + evidence, pH, protonation state, net charge,
charge method, force field, planned water model, random seed, per-step
status/timing, and tool versions.

**GPU side then runs:** `tleap -f leap_solvate.in` → solvated `*_solv.prmtop/rst7`
→ OpenMM equilibration + production. **Returned for analysis:** trajectory
(NetCDF/DCD) + topology (prmtop) → RMSD alignment, PCA, clustering → the
conformational ensemble.

## Running

```bash
conda activate mdsprep
python scripts/02_prep_molecule.py --from-validation T1_smoke      # one molecule
python scripts/03_prep_batch.py --input data/validation_subset.csv # fail-soft batch
python tests/test_prep.py                                          # offline checks
```

The batch driver skips failures (and unparseable peptides) and writes
`batch_report.{csv,json}` so one bad molecule never halts a 100-molecule run.
