"""Offline tests for Stage-3 viz/features (no MD trajectory needed)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ensemble_qsar.features import aggregate  # noqa: E402
from ensemble_qsar.viz import build_html, render3d  # noqa: E402

_VIZ = {
    "meta": {"mol_id": "demo", "smiles": "CCO", "n_frames": 3,
             "equilibration_frame": 0, "save_interval_ps": 0.4, "versions": {}},
    "frames": [
        {"frame": 0, "time_ps": 0.0, "tsne1": 1.0, "tsne2": 2.0, "pc1": 0.1, "pc2": 0.2,
         "cluster": 0, "psa3d": 20.0, "rg": 3.0, "intra_hbond": 0, "sasa": 200.0, "rmsd": 0.1},
        {"frame": 1, "time_ps": 0.4, "tsne1": 1.1, "tsne2": 2.1, "pc1": 0.2, "pc2": 0.3,
         "cluster": 1, "psa3d": 22.0, "rg": 3.2, "intra_hbond": 1, "sasa": 210.0, "rmsd": 0.2},
        {"frame": 2, "time_ps": 0.8, "tsne1": 0.9, "tsne2": 1.9, "pc1": 0.0, "pc2": 0.1,
         "cluster": 0, "psa3d": 21.0, "rg": 3.1, "intra_hbond": 0, "sasa": 205.0, "rmsd": 0.15},
    ],
    "clusters": [
        {"id": 0, "population": 2, "rep_frame": 0, "images": ["data:image/png;base64,AAA"],
         "mean_metrics": {"psa3d": 20.5, "rg": 3.05, "intra_hbond": 0, "sasa": 202.5, "rmsd": 0.12}},
        {"id": 1, "population": 1, "rep_frame": 1, "images": ["data:image/png;base64,BBB"],
         "mean_metrics": {"psa3d": 22.0, "rg": 3.2, "intra_hbond": 1, "sasa": 210.0, "rmsd": 0.2}},
    ],
    "summary": {"psa3d": {"mean": 21.0, "std": 0.8, "min": 20.0, "max": 22.0,
                          "q25": 20.5, "q50": 21.0, "q75": 21.5},
                "ensemble_weighted": {"psa3d": 21.0}},
    "descriptor_definitions": {},
}


def test_molecule_features_flatten():
    row = aggregate.molecule_features(_VIZ)
    assert row["mol_id"] == "demo" and row["n_clusters"] == 2
    assert row["psa3d_mean"] == 21.0 and row["psa3d_ensw"] == 21.0


def test_features_table_label_join():
    df = aggregate.build_features_table([_VIZ], stage1_labels={"demo": -5.2})
    assert list(df["mol_id"]) == ["demo"]
    assert df.loc[0, "label"] == -5.2


def test_build_html_self_contained(tmp_path=Path("/tmp")):
    out = Path("/tmp/_viz_test.html")
    build_html.build_html(_VIZ, out)
    html = out.read_text()
    assert "__VIZ_DATA__" not in html          # placeholder replaced
    assert "const DATA = {" in html            # data injected
    assert "http://" not in html and "https://" not in html  # self-contained
    out.unlink()


def test_pdb_fallback_parses_coords():
    pdb = ("ATOM      1  C   MOL A   1       0.000   0.000   0.000  1.00  0.00           C\n"
           "ATOM      2  O   MOL A   1       1.200   0.000   0.000  1.00  0.00           O\n")
    p = Path("/tmp/_tiny.pdb"); p.write_text(pdb)
    xyz, sym, bonds = render3d._parse_pdb_fallback(p)
    assert len(xyz) == 2 and sym == ["C", "O"] and (0, 1) in bonds
    p.unlink()


def test_hub_thumbnail_and_build():
    import json
    import shutil
    from ensemble_qsar.viz import build_hub

    svg = build_hub.render_thumbnail("CCO")
    assert svg.startswith("<svg") and "</svg>" in svg
    assert build_hub.render_thumbnail("not_a_smiles") == ""

    root = Path("/tmp/_hubtest")
    shutil.rmtree(root, ignore_errors=True)
    done = root / "molA"; done.mkdir(parents=True)
    (done / "run_manifest.json").write_text(json.dumps(
        {"prep_manifest": {"mol_id": "molA", "mol_class": "small_molecule"}}))
    (done / "viz_data.json").write_text(json.dumps({
        "meta": {"mol_id": "molA", "smiles": "CCO", "n_frames": 10, "equilibration_frame": 0},
        "clusters": [{"id": 0}],
        "summary": {m: {"mean": 1.0, "std": 0.1} for m in
                    ("psa3d", "rg", "intra_hbond", "sasa", "rmsd")}}))
    pend = root / "molB"; pend.mkdir()
    (pend / "manifest.json").write_text(json.dumps({"mol_id": "molB", "mol_class": "peptide"}))

    hub = build_hub.collect_hub_index(root)
    assert hub["n_molecules"] == 1 and hub["n_pending"] == 1
    assert hub["molecules"][0]["mw"] is not None  # MW computed from SMILES

    html_path, _ = build_hub.build_hub(root)
    html = html_path.read_text()
    assert "__HUB_DATA__" not in html               # data injected
    assert 'src="http' not in html and 'href="http' not in html  # no external loads
    shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    test_molecule_features_flatten()
    test_features_table_label_join()
    test_build_html_self_contained()
    test_pdb_fallback_parses_coords()
    test_hub_thumbnail_and_build()
    print("all viz tests passed")
