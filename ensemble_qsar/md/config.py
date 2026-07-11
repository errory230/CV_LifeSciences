"""Stage 2b configuration — all tunables in one place (never hard-coded).

The notebook's top CONFIG cell builds an `MDConfig`; every step reads from it.
Values that the prep stage already decided (water model, pH, random seed) are
merged in from the molecule's `manifest.json` so the two stages stay consistent.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class MDConfig:
    # --- solvation ----------------------------------------------------------
    water_model: str = "tip3p"            # overridden by prep manifest
    solvation_engine: str = "tleap"       # "tleap" (uses prep's leap_solvate.in)

    # --- integrator / thermodynamics ---------------------------------------
    temperature_K: float = 300.0
    pressure_bar: float = 1.0
    timestep_fs: float = 4.0              # 4 fs is safe with HMR + HBonds
    hmr: bool = True                      # hydrogen-mass repartitioning
    hydrogen_mass_amu: float = 3.5        # H mass for HMR (enables 4 fs)
    friction_ps: float = 1.0
    nonbonded_cutoff_nm: float = 1.0
    constraints: str = "HBonds"           # HBonds | AllBonds | None

    # --- minimization -------------------------------------------------------
    minimize_max_iter: int = 0            # 0 = minimize to convergence

    # --- equilibration ------------------------------------------------------
    nvt_ns: float = 0.1
    npt_ns: float = 0.2
    # restraint force constant (kcal/mol/A^2) on solute heavy atoms, released
    # in stages across NPT so the solute relaxes gradually.
    restraint_schedule: tuple[float, ...] = (10.0, 5.0, 2.0, 0.0)

    # --- production ---------------------------------------------------------
    production_ns: float = 1.0            # TEST value; raise to 50 for real runs
    save_interval_ps: float = 10.0        # trajectory frame spacing
    checkpoint_interval_ps: float = 100.0 # checkpoint-to-Drive spacing

    # --- platform -----------------------------------------------------------
    platform: str = "CUDA"                # CUDA on Colab; CPU for local dry-run
    precision: str = "mixed"
    seed: int = 42                        # overridden by prep manifest

    # --- analysis -----------------------------------------------------------
    pca_components: int = 10
    n_clusters: int = 5
    cluster_on: str = "pca"               # "pca" | "rmsd"
    tsne: bool = True                     # add a t-SNE embedding of the ensemble
    tsne_perplexity: float = 30.0         # auto-capped to the frame count
    # per-frame descriptors to compute; psa3d + rmsd are the permeability core.
    descriptors: tuple[str, ...] = ("psa3d", "rmsd")

    # --- io -----------------------------------------------------------------
    input_root: str = "data/prep"
    output_root: str = "data/md"
    use_drive: bool = False

    def merge_manifest(self, manifest: dict) -> "MDConfig":
        """Return a copy with prep-decided fields taken from the manifest."""
        wm = manifest.get("water_model_planned") or self.water_model
        seed = manifest.get("random_seed")
        seed = int(seed) if seed is not None else self.seed
        return _replace(self, water_model=wm, seed=seed)

    def n_steps(self, ns: float) -> int:
        """Convert a duration in ns to an integer step count at this timestep."""
        return int(round(ns * 1e6 / self.timestep_fs))

    def save_interval_steps(self) -> int:
        return max(1, int(round(self.save_interval_ps * 1e3 / self.timestep_fs)))

    def checkpoint_interval_steps(self) -> int:
        return max(1, int(round(self.checkpoint_interval_ps * 1e3 / self.timestep_fs)))

    def as_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2))


def _replace(cfg: MDConfig, **changes) -> MDConfig:
    data = asdict(cfg)
    data.update(changes)
    # tuples survive asdict as lists; restore where the field expects a tuple
    for k in ("restraint_schedule", "descriptors"):
        if isinstance(data.get(k), list):
            data[k] = tuple(data[k])
    return MDConfig(**data)
