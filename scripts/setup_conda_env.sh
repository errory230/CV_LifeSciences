#!/usr/bin/env bash
# Bootstrap the CPU MD-prep conda environment (Stage 2a).
#
# Installs Miniconda if it is not already present, points conda at the agent
# proxy's CA bundle so package downloads verify correctly, then creates the
# `mdsprep` environment from environment.yml.
#
# Usage:  bash scripts/setup_conda_env.sh
# Then:   conda activate mdsprep   (or use "$CONDA_DIR/envs/mdsprep/bin/python")
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
CA="${CCR_CA_BUNDLE:-/root/.ccr/ca-bundle.crt}"

# Make every TLS client in this script trust the proxy CA.
if [ -f "$CA" ]; then
  export REQUESTS_CA_BUNDLE="$CA" SSL_CERT_FILE="$CA" CURL_CA_BUNDLE="$CA"
fi

if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  echo ">> installing Miniconda into $CONDA_DIR"
  curl -fsSL -o /tmp/miniconda.sh \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
fi

# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"

# conda ships its own certifi bundle; redirect it to the proxy CA.
if [ -f "$CA" ]; then
  conda config --system --set ssl_verify "$CA"
fi
# Use ONLY conda-forge: drop the Anaconda default channels (avoids their ToS
# gate and keeps every dependency from a single community channel).
conda config --system --remove-key channels 2>/dev/null || true
conda config --system --add channels conda-forge
conda config --system --set channel_priority strict

if conda env list | grep -qE '^\s*mdsprep\s'; then
  echo ">> updating existing mdsprep env"
  conda env update -n mdsprep -f "$ROOT/environment.yml" --prune
else
  echo ">> creating mdsprep env (this can take several minutes)"
  conda env create -f "$ROOT/environment.yml"
fi

echo ">> verifying key tools"
"$CONDA_DIR/envs/mdsprep/bin/python" - <<'PY'
import shutil, subprocess, sys
import rdkit, parmed  # noqa: F401
for exe in ("antechamber", "parmchk2", "tleap", "sqm"):
    path = shutil.which(exe)
    print(f"  {exe}: {path or 'MISSING'}")
print("  rdkit:", rdkit.__version__)
print("OK")
PY
echo ">> done. Activate with: conda activate mdsprep"
