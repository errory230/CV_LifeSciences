# MDS-Ensemble QSAR

**Conformational-ensemble descriptors for flexible molecules — a molecular-dynamics
approach to QSAR.**

## Motivation

Classical QSAR represents a molecule by descriptors computed from a **single,
static structure**. For small, rigid drug-like molecules that is a reasonable
approximation. But molecules larger than the classic "rule of 5" chemical space —
macrocycles, cyclic peptides, long flexible chains, many natural products —
are not static: in solution they wobble between **many conformations**, much like
an intrinsically disordered protein (IDP) samples an ensemble rather than a fixed
fold. For properties that depend on 3D shape and solvent exposure
(permeability, solubility, and ultimately bioavailability), collapsing that
ensemble to one conformer throws away exactly the information that matters.

**Hypothesis.** Describing a flexible molecule by a *conformational ensemble*
generated with molecular-dynamics simulation (MDS) in explicit solvent yields
QSAR features that are more predictive than single-conformer features — and the
gain grows with molecular flexibility.

## Approach

```
 public property data ──▶ curate ──▶ flexibility descriptors ──▶ representative set
                                                                        │
                                                                        ▼
                                        MD in solvent (water, set pH) ──▶ conformer
                                                                        ensemble
                                                                        │
                                                                        ▼
                          ensemble-averaged descriptors ──▶ QSAR ──▶ compare vs static
```

We prove the concept on two **data-rich public benchmarks** — one per property
the project targets:

| Dataset      | Property (endpoint)      | Source                          | n (curated) |
|--------------|--------------------------|---------------------------------|-------------|
| `caco2_wang` | Caco-2 permeability      | Wang et al. 2016 / TDC          | ~896        |
| `esol`       | Aqueous solubility       | Delaney 2004 / MoleculeNet ESOL | ~1117       |

These two datasets are also a natural **contrast**: `caco2_wang` is rich in
flexible, beyond-rule-of-5 molecules (≈261 bRo5, ≈81 macrocycles), where the
ensemble hypothesis should bite; `esol` is dominated by small rigid molecules,
a baseline where a single conformer should already suffice.

## Stage 1 — data extraction (implemented)

`scripts/01_build_poc_dataset.py` runs the full stage-1 pipeline:

1. **Acquire** — download each benchmark from a pinned, checksum-verified mirror
   (`ensemble_qsar/data/sources.py`). Cached under `data/raw/`.
2. **Curate** — RDKit standardization: largest organic fragment, neutralize,
   canonical SMILES + InChIKey, drop non-organic/unparseable, and collapse
   duplicate structures (label aggregated by median). Every drop is counted in
   an auditable `CurationReport` (`ensemble_qsar/data/curate.py`).
3. **Describe** — physicochemical + **flexibility** descriptors: rotatable
   bonds, Kier flexibility index, macrocycle flag, Fsp3, TPSA, cLogP, HBD/HBA,
   rule-of-5 violations, and a coarse `flex_class`
   (`ensemble_qsar/data/descriptors.py`).
4. **Select** — a flexibility-stratified, chemically-diverse **representative
   set** for the MD stage: MaxMin diversity picks within rotatable-bond strata,
   deliberately over-sampling the flexible tail while keeping a rigid control
   group (`ensemble_qsar/data/select.py`).

Outputs land in `data/processed/<dataset>/`. The small stage-1 deliverables —
`representative_set.csv` and `summary.json` — are committed as a snapshot; the
full `curated.{csv,parquet}` tables are regenerable build artefacts (rerun the
script) and are gitignored.

### Run it

```bash
pip install -r requirements.txt
python scripts/01_build_poc_dataset.py                 # both datasets
python scripts/01_build_poc_dataset.py --datasets caco2_wang --n-select 80
python tests/test_pipeline.py                          # offline sanity checks
```

## Stage 2a — CPU MD prep (implemented)

Turns each selected molecule into a **parameterized, GPU-ready MD input** using
only CPU — conformer generation, pH protonation, AM1-BCC/GAFF2 (or ff19SB for
peptides) parameterization — and hands off a dry system plus a `leap_solvate.in`
recipe for the GPU stage. Class-aware routing (peptide vs small molecule) with
review flags. See [`docs/PREP.md`](docs/PREP.md).

```bash
bash scripts/setup_conda_env.sh && conda activate mdsprep
python scripts/02_prep_molecule.py --from-validation T1_smoke
python scripts/03_prep_batch.py --input data/validation_subset.csv
```

## Stage 2b — GPU MD + ensemble analysis (implemented, Colab)

`notebooks/stage2b_md_colab.ipynb` (thin driver over `ensemble_qsar/md/`) takes
the prep hand-off and runs **solvate → minimize → equilibrate → production →
analysis** on a Colab GPU, yielding each molecule's conformational ensemble with
**3D-PSA**/RMSD, PCA and cluster representatives. Checkpoint/resume-safe across
Colab disconnects; fail-soft batch. Logic validated end-to-end on CPU OpenMM.

## Roadmap

See [`docs/ROADMAP.md`](docs/ROADMAP.md). Stage 1 (data), Stage 2a (CPU prep) and
Stage 2b (GPU MD + ensemble analysis) are done; next is aggregating ensemble
descriptors into QSAR models and the ensemble-vs-static comparison (Stage 3–4).

## Repository layout

```
ensemble_qsar/data/   sources · curate · descriptors · select
scripts/              01_build_poc_dataset.py  (stage-1 orchestrator)
data/processed/       curated datasets + representative sets (committed)
data/raw/             cached mirror downloads (gitignored)
tests/                offline pipeline sanity tests
docs/ROADMAP.md       staged plan
```

## Data provenance & licensing

Datasets are redistributed from their public mirrors solely as a reproducible
convenience; original terms apply. Caco-2: Wang et al., *J. Chem. Inf. Model.*
2016. ESOL: Delaney, *J. Chem. Inf. Comput. Sci.* 2004 (MoleculeNet).
