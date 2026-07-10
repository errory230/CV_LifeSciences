"""Offline tests for the prep stage that need no conda tools (RDKit only).

Classification and peptide-sequence parsing are pure-RDKit/string logic; the
external-tool steps (antechamber/tleap) are exercised by the T-tier runs, not
here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ensemble_qsar.prep import classify, peptide, protonate  # noqa: E402


def test_classify_routes():
    # true linear peptide -> peptide route
    pep = classify.classify_molecule(
        "CC(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](C)C(=O)NCC(=O)NCC(=O)"
        "N[C@@H](CCCCN)C(=O)N[C@@H](C)C(N)=O")
    assert pep.mol_class == "peptide"
    assert pep.n_residues >= 3

    # everolimus: macrocyclic polyketide, NOT a peptide -> small molecule + review
    sdz = classify.classify_molecule(
        "CO[C@H]1C[C@@H]2CC[C@@H](C)[C@@](O)(O2)C(=O)C(=O)N2CCCC3[C@H]2C(=O)O"
        "[C@@H](CC(=O)[C@H](C)/C=C(\\C)[C@@H](O)[C@@H](OC)C(=O)[C@H](C)CC[C@H]"
        "(C)/C=C/C=C/C=C/1C)[C@H]3C[C@@H]1CC[C@@H](OCCO)[C@H](OC)C1")
    assert sdz.mol_class == "small_molecule"
    assert sdz.is_macrocycle and sdz.needs_review

    # pyridine -> small molecule, no review
    pyr = classify.classify_molecule("c1ccncc1")
    assert pyr.mol_class == "small_molecule" and not pyr.needs_review


def test_peptide_sequence_from_name():
    r = peptide.sequence_from_name("Ac-Trp-Ala-Gly-Gly-Lys-Ala-NH2")
    assert r.ok
    assert r.residues == ["ACE", "TRP", "ALA", "GLY", "GLY", "LYS", "ALA", "NHE"]

    bad = peptide.sequence_from_name("Bis(7)-tacrine")
    assert not bad.ok


def test_protonation_rejects_amidate():
    # T2: dimorphite must not hand back a deprotonated amide at pH 7.4
    smi = ("Cc1cc(COc2ccc(C(=O)N[C@@H]3CC4(C[C@@H]3C(=O)NO)OCCO4)cc2)c2ccccc2n1")
    res = protonate.protonate_small_molecule(smi, ph=7.4)
    assert res.net_charge == 0
    assert "[N-]" not in res.smiles


if __name__ == "__main__":
    test_classify_routes()
    test_peptide_sequence_from_name()
    test_protonation_rejects_amidate()
    print("all prep tests passed")
