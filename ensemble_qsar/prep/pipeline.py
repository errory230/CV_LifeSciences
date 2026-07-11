"""Per-molecule prep orchestrator (CPU, GPU-handoff producer).

Runs the staged pipeline for one molecule, writing all artefacts under
`<out_root>/<mol_id>/` and tracking per-step status in the manifest so a re-run
resumes rather than repeats. External-tool steps run inside an `intermediates/`
working directory (keeping debug files), and the canonical handoff files are
copied up to the molecule directory.

Small-molecule route (GAFF2 + AM1-BCC) is implemented. The peptide route is a
declared stub added in a later stage; the batch driver records it as skipped.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem

from . import charges, classify, conformer, parameterize, peptide, protonate
from ._util import tool_version
from .manifest import Manifest, StepRecord


@dataclass
class PrepConfig:
    ph: float = 7.4
    random_seed: int = 42
    n_conformers: int = 20
    water_model: str = "tip3p"      # planned solvent for the GPU stage
    box_padding: float = 12.0       # Angstrom, written into the solvate recipe
    atom_type: str = "gaff2"


# canonical small-molecule files copied from intermediates/ up to the mol dir
_HANDOFF = [
    "ligand.mol2", "ligand.frcmod", "ligand_gas.prmtop", "ligand_gas.rst7",
    "ligand_ref.pdb", "leap_solvate.in",
]
# canonical peptide-route handoff files
_HANDOFF_PEPTIDE = [
    "peptide_gas.prmtop", "peptide_gas.rst7", "peptide_capped.pdb", "leap_solvate.in",
]


class PeptideSequenceUnknown(RuntimeError):
    """Raised when a peptide's sequence cannot be parsed (flag for review)."""


def prep_molecule(
    mol_id: str, smiles: str, *, out_root: Path, config: PrepConfig | None = None
) -> Manifest:
    cfg = config or PrepConfig()
    # Resolve to absolute: the external tools (antechamber/parmchk2) run with
    # cwd set to the intermediates dir, so any relative input/output path passed
    # to them would break. Absolute paths make the route cwd-independent.
    out_root = Path(out_root).resolve()
    mol_dir = out_root / _safe(mol_id)
    work = mol_dir / "intermediates"
    work.mkdir(parents=True, exist_ok=True)

    m = Manifest.load_or_new(mol_id, smiles, mol_dir)
    m.random_seed = cfg.random_seed
    m.water_model_planned = cfg.water_model
    m.tool_versions = {t: tool_version(t) for t in ("antechamber", "parmchk2", "tleap", "sqm")}

    # --- Step 1: classify ---------------------------------------------------
    cls = classify.classify_molecule(smiles)
    m.mol_class = cls.mol_class
    m.class_evidence = cls.evidence
    m.needs_review = cls.needs_review
    m.record(StepRecord("classify", "ok", info=cls.as_dict()))

    if cls.mol_class == "peptide":
        return _prep_peptide(m, mol_id, smiles, mol_dir, work, cfg, cls)

    # --- Step 2: protonation at pH -----------------------------------------
    if not m.is_done("protonate"):
        prot = protonate.protonate_small_molecule(smiles, ph=cfg.ph)
        m.ph = prot.ph
        m.protonation_state = prot.smiles
        m.net_charge = prot.net_charge
        m.charge_method = "AM1-BCC (antechamber/sqm)"
        m.record(StepRecord("protonate", "ok", info={
            "tool": prot.tool, "state": prot.smiles, "net_charge": prot.net_charge,
            "variants": prot.all_variants, "note": prot.note}))
    prot_smiles = m.protonation_state
    net_charge = int(m.net_charge or 0)

    # --- Step 3: conformer + minimization ----------------------------------
    sdf = work / "conformer.sdf"
    if not (m.is_done("conformer") and sdf.exists()):
        conf = conformer.generate_conformer(
            prot_smiles, random_seed=cfg.random_seed, n_conformers=cfg.n_conformers)
        conformer.write_sdf(conf.mol, str(sdf))
        m.forcefield = "GAFF2"
        m.record(StepRecord("conformer", "ok", seconds=0.0, outputs=[str(sdf)], info={
            "energy": conf.energy, "ff": conf.forcefield,
            "tried": conf.n_conformers_tried, "converged": conf.n_conformers_converged}))

    # --- Step 4: AM1-BCC charges -------------------------------------------
    if not m.is_done("charges"):
        ch = charges.assign_am1bcc_charges(
            sdf, out_dir=work, net_charge=net_charge, atom_type=cfg.atom_type)
        fallback = False
        if not ch.ok:
            # Safety net: a protonation microstate can be antechamber-hostile
            # (e.g. an over-deprotonated N). Retry once with the neutral parent.
            neutral = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
            conf = conformer.generate_conformer(
                neutral, random_seed=cfg.random_seed, n_conformers=cfg.n_conformers)
            conformer.write_sdf(conf.mol, str(sdf))
            ch = charges.assign_am1bcc_charges(
                sdf, out_dir=work, net_charge=0, atom_type=cfg.atom_type)
            if ch.ok:
                fallback = True
                net_charge = 0
                m.net_charge = 0
                m.protonation_state = f"{neutral} (neutral fallback)"
        if not ch.ok:
            m.record(StepRecord("charges", "failed", seconds=ch.cmd.seconds,
                                info={"stderr_tail": ch.cmd.stderr[-800:]}))
            m.save()
            raise RuntimeError(f"{mol_id}: AM1-BCC charge assignment failed")
        m.record(StepRecord("charges", "ok", seconds=ch.cmd.seconds,
                            outputs=[str(ch.mol2_path)],
                            info={"method": ch.charge_method, "neutral_fallback": fallback}))

    # --- Step 5: GAFF2 parameterization + handoff --------------------------
    if not m.is_done("parameterize"):
        par = parameterize.parameterize_gaff2(
            work / "ligand.mol2", out_dir=work, net_charge=net_charge,
            water_model=cfg.water_model, box_padding=cfg.box_padding)
        status = "ok" if par.ok else "failed"
        m.record(StepRecord("parameterize", status, outputs=[
            str(par.frcmod_path), str(par.gas_prmtop), str(par.solvate_recipe)],
            info={"messages": par.messages}))
        if not par.ok:
            m.save()
            raise RuntimeError(f"{mol_id}: parameterization failed ({par.messages})")

    # --- copy handoff files up + record ------------------------------------
    handoff: list[str] = []
    for name in _HANDOFF:
        src = work / name
        if src.exists():
            dst = mol_dir / name
            shutil.copy2(src, dst)
            handoff.append(name)
    # reference SDF alongside handoff
    if sdf.exists():
        shutil.copy2(sdf, mol_dir / "ligand.sdf")
        handoff.append("ligand.sdf")
    m.handoff_files = handoff
    m.save()
    return m


def _prep_peptide(m, mol_id, smiles, mol_dir, work, cfg, cls) -> Manifest:
    """Peptide route: ff19SB library-charge build from the parsed sequence."""
    m.forcefield = "ff19SB"
    m.charge_method = "protein force-field library charges"
    m.ph = cfg.ph

    seq = peptide.sequence_from_name(mol_id)
    if not seq.ok:
        m.record(StepRecord("route", "skipped", info={
            "reason": "peptide sequence not parseable from name",
            "detail": seq.notes,
            "todo": "automatic SMILES->sequence perception is future work"}))
        m.needs_review = True
        m.save()
        raise PeptideSequenceUnknown(f"{mol_id}: {seq.notes}")

    m.record(StepRecord("sequence", "ok", info={"residues": seq.residues, "notes": seq.notes}))
    m.protonation_state = (
        "ff19SB standard states @ pH 7.4 (LYS+/ARG+/ASP-/GLU-/HIS=HIE)")

    if not m.is_done("parameterize"):
        res = peptide.build_peptide(
            seq.residues, out_dir=work, reference_smiles=smiles,
            cyclic=cls.is_macrocycle, water_model=cfg.water_model,
            box_padding=cfg.box_padding)
        status = "ok" if res.ok else "failed"
        m.net_charge = res.net_charge
        m.record(StepRecord("parameterize", status, outputs=[
            str(res.gas_prmtop), str(res.solvate_recipe)],
            info={"residues": res.residues, "net_charge": res.net_charge,
                  "cyclic": cls.is_macrocycle, "messages": res.messages}))
        if not res.ok:
            m.save()
            raise RuntimeError(f"{mol_id}: peptide build failed ({res.messages})")

    handoff: list[str] = []
    for name in _HANDOFF_PEPTIDE:
        src = work / name
        if src.exists():
            shutil.copy2(src, mol_dir / name)
            handoff.append(name)
    m.handoff_files = handoff
    m.save()
    return m


def _safe(mol_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in mol_id)
