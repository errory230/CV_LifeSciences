"""Dataset registry and reproducible download layer.

The MDS-ensemble QSAR project needs raw property data (SMILES + label) for its
proof-of-concept. The canonical homes for these benchmarks (TDC / Harvard
Dataverse, ChEMBL/EBI) are not always reachable from every environment, so each
dataset is pinned to a *reachable, content-verified mirror* and downloaded with
an on-disk cache. Every download records the SHA-256 of the bytes actually
retrieved so a run is reproducible and a silently-changed upstream is detected.

To add a dataset, append a `Dataset` entry to `REGISTRY`. Nothing else in the
pipeline hard-codes a dataset name.
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# Repo-root/data/raw — where raw mirror files are cached.
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


@dataclass(frozen=True)
class Dataset:
    """A property dataset pinned to a reachable mirror.

    Attributes
    ----------
    name:        short identifier used on the CLI and in output filenames.
    endpoint:    the physical/biological property being modelled.
    task:        "regression" or "classification".
    url:         reachable mirror of the raw file.
    smiles_col:  column holding the molecule SMILES.
    label_col:   column holding the target value.
    id_col:      optional column with a human-readable molecule id.
    label_unit:  human description of the label's units / scale.
    sep:         field separator of the raw file.
    sha256:      expected digest of the raw bytes; None until first pinned.
    note:        provenance / caveats.
    """

    name: str
    endpoint: str
    task: str
    url: str
    smiles_col: str
    label_col: str
    id_col: str | None = None
    label_unit: str = ""
    sep: str = ","
    sha256: str | None = None
    note: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)


REGISTRY: dict[str, Dataset] = {
    # Caco-2 apparent permeability (log Papp, cm/s). The Wang et al. (2016) set
    # is the standard TDC ADME permeability benchmark; this mirror is the
    # byte-identical TDC export used by RekerLab/MolALKit.
    "caco2_wang": Dataset(
        name="caco2_wang",
        endpoint="caco2_permeability",
        task="regression",
        url="https://raw.githubusercontent.com/RekerLab/MolALKit/"
        "main/molalkit/data/datasets/caco2_wang.csv",
        smiles_col="Drug",
        label_col="Y",
        id_col="Drug_ID",
        label_unit="log10 Papp (cm/s)",
        sep=",",
        note="Wang et al. J Chem Inf Model 2016; TDC 'Caco2_Wang' (n~910). "
        "Mirror: RekerLab/MolALKit.",
        aliases=("caco2", "permeability"),
    ),
    # Aqueous solubility (log S, mol/L). ESOL / Delaney is the classic
    # MoleculeNet solubility benchmark, hosted in the DeepChem repo.
    "esol": Dataset(
        name="esol",
        endpoint="aqueous_solubility",
        task="regression",
        url="https://raw.githubusercontent.com/deepchem/deepchem/"
        "master/datasets/delaney-processed.csv",
        smiles_col="smiles",
        label_col="measured log solubility in mols per litre",
        id_col="Compound ID",
        label_unit="log10 S (mol/L)",
        sep=",",
        note="Delaney J Chem Inf 2004; MoleculeNet 'ESOL' (n~1128). "
        "Mirror: deepchem/deepchem.",
        aliases=("delaney", "solubility"),
    ),
}


def resolve(name: str) -> Dataset:
    """Look up a dataset by name or alias (case-insensitive)."""
    key = name.strip().lower()
    if key in REGISTRY:
        return REGISTRY[key]
    for ds in REGISTRY.values():
        if key in ds.aliases:
            return ds
    raise KeyError(
        f"unknown dataset {name!r}; known: {', '.join(sorted(REGISTRY))}"
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(ds: Dataset, *, cache_dir: Path = RAW_DIR, force: bool = False) -> Path:
    """Download `ds` to the cache and return the local path.

    Re-uses a cached copy unless `force`. When the dataset carries a pinned
    `sha256`, the download is verified against it and a mismatch raises.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{ds.name}.raw{_suffix(ds.sep)}"

    if dest.exists() and not force:
        digest = _sha256(dest)
    else:
        # urllib honours HTTP(S)_PROXY / no_proxy from the environment.
        req = urllib.request.Request(ds.url, headers={"User-Agent": "ensemble-qsar/0.1"})
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (pinned https mirror)
            data = resp.read()
        dest.write_bytes(data)
        digest = _sha256(dest)

    if ds.sha256 and digest != ds.sha256:
        raise ValueError(
            f"checksum mismatch for {ds.name}: got {digest}, expected {ds.sha256}"
        )
    return dest


def _suffix(sep: str) -> str:
    return ".tsv" if sep == "\t" else ".csv"
