"""Stage 2b: GPU MD (OpenMM) + ensemble analysis.

Designed for Google Colab but engine-agnostic: the notebook is a thin driver
over these functions, which also run on a CPU OpenMM platform for local dry-runs.
"""

# Only the OpenMM-free submodules are imported eagerly so that Stage-3 analysis
# (which needs md.analyze's helpers but no simulation engine) can run without
# OpenMM installed. simulate / solvate / pipeline import OpenMM and are loaded on
# demand via explicit `from ensemble_qsar.md import simulate` etc.
from . import analyze, config, io

__all__ = ["config", "io", "analyze", "solvate", "simulate", "pipeline"]
