"""Stage-3 QSAR features: aggregate a molecule's ensemble into one feature row.

The MD ensemble (many frames) is collapsed to a per-molecule feature vector — the
distribution statistics of each ensemble descriptor — which is the input a QSAR
model consumes. Optionally the Stage-1 label (and static 2D descriptors) are
joined by `mol_id`, giving the table needed to compare static-vs-ensemble models.

`molecule_features` -> flat dict (one row); `build_features_table` -> DataFrame
across molecules; `write_feature_detail` -> the richer per-molecule JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_METRICS = ["psa3d", "rg", "intra_hbond", "sasa", "rmsd"]
_STATS = ["mean", "std", "min", "max", "q25", "q50", "q75"]


def molecule_features(viz: dict) -> dict:
    """Flatten one molecule's ensemble summary into a feature row."""
    row = {"mol_id": viz["meta"]["mol_id"], "smiles": viz["meta"].get("smiles", ""),
           "n_frames": viz["meta"]["n_frames"], "n_clusters": len(viz["clusters"])}
    summ = viz["summary"]
    for m in _METRICS:
        if m in summ:
            for s in _STATS:
                row[f"{m}_{s}"] = summ[m].get(s)
        w = summ.get("ensemble_weighted", {}).get(m)
        if w is not None:
            row[f"{m}_ensw"] = w
    return row


def write_feature_detail(viz: dict, out_path: Path) -> Path:
    """Per-molecule detail: summary + per-cluster means + descriptor definitions."""
    detail = {
        "mol_id": viz["meta"]["mol_id"],
        "smiles": viz["meta"].get("smiles", ""),
        "n_frames": viz["meta"]["n_frames"],
        "summary": viz["summary"],
        "clusters": [{k: c[k] for k in ("id", "population", "rep_frame", "mean_metrics")}
                     for c in viz["clusters"]],
        "descriptor_definitions": viz["descriptor_definitions"],
    }
    Path(out_path).write_text(json.dumps(detail, indent=2))
    return Path(out_path)


def build_features_table(viz_list: list[dict], *, stage1_labels: dict | None = None,
                         static_desc: dict | None = None) -> pd.DataFrame:
    """One row per molecule; optionally join Stage-1 label + static descriptors.

    stage1_labels / static_desc: {mol_id: value} / {mol_id: {desc: value}} keyed
    by the same mol_id carried through from Stage 1.
    """
    rows = [molecule_features(v) for v in viz_list]
    df = pd.DataFrame(rows)
    if static_desc:
        sd = pd.DataFrame(static_desc).T.add_prefix("static_")
        df = df.merge(sd, left_on="mol_id", right_index=True, how="left")
    if stage1_labels:
        df["label"] = df["mol_id"].map(stage1_labels)
    return df
