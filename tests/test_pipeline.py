"""Offline sanity tests for the stage-1 data pipeline (no network)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ensemble_qsar.data import curate, descriptors, select  # noqa: E402


def _toy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["aspirin", "aspirin_salt", "bad", "caffeine", "flexible"],
            "smi": [
                "CC(=O)Oc1ccccc1C(=O)O",
                "CC(=O)Oc1ccccc1C(=O)[O-].[Na+]",  # salt -> same parent as aspirin
                "not_a_smiles",
                "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
                "CCCCCCCCCCCCCCCCO",  # long-chain alcohol, very flexible
            ],
            "y": [-2.1, -2.2, 9.9, -1.0, -3.5],
        }
    )


def test_curation_dedup_and_salt_stripping():
    curated, report = curate.curate_frame(
        _toy_frame(), smiles_col="smi", label_col="y", id_col="id"
    )
    # aspirin and its sodium salt collapse to one parent; the junk SMILES drops.
    assert report.n_unparseable == 1
    keys = set(curated["inchikey"])
    assert len(keys) == 3  # aspirin, caffeine, flexible alcohol
    # the only structure seen twice (neutral + salt) must be merged
    assert curated["n_measurements"].max() == 2
    assert (curated["n_measurements"] == 2).sum() == 1


def test_descriptor_flexibility_ordering():
    curated, _ = curate.curate_frame(
        _toy_frame(), smiles_col="smi", label_col="y", id_col="id"
    )
    annotated = descriptors.add_descriptors(curated)
    long_chain = annotated[annotated["smiles"] == "CCCCCCCCCCCCCCCCO"].iloc[0]
    caffeine = annotated[annotated["mw"].between(190, 200)].iloc[0]
    assert long_chain["n_rotatable_bonds"] > caffeine["n_rotatable_bonds"]
    assert long_chain["flex_class"] == "very_flexible"
    assert caffeine["flex_class"] == "rigid"


def test_selection_respects_budget():
    df = pd.DataFrame(
        {
            "smiles": [
                "C" * 0 + s
                for s in [
                    "CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCCCCCCCO",
                    "CCCCCCCCCCN", "c1ccc2ccccc2c1", "CCOCCOCCOCC",
                ]
            ],
        }
    )
    df = descriptors.add_descriptors(df)
    picked = select.select_representative_set(df, n_select=4, seed=0)
    assert len(picked) == 4
    assert "selection_stratum" in picked.columns


if __name__ == "__main__":
    test_curation_dedup_and_salt_stripping()
    test_descriptor_flexibility_ordering()
    test_selection_respects_budget()
    print("all stage-1 tests passed")
