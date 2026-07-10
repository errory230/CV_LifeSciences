"""Steps 2-4: OpenMM minimization, equilibration, production (GPU on Colab).

Each stage builds its own `Simulation` from files on disk so it can run — and
resume — independently:

  minimize      -> minimized_state.xml
  equilibrate   -> equilibrated.chk   (NVT heat, then NPT with restraint release)
  produce       -> trajectory.dcd + production.chk + statedata.csv

Production is checkpoint/resume-safe: on re-entry it reloads production.chk and
the progress file and continues appending to the trajectory, so a dropped Colab
session does not restart the run.

Restraints on solute heavy atoms use a CustomExternalForce whose stiffness `k`
is a global parameter, stepped down across the schedule in cfg.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import openmm
from openmm import app, unit

# kcal/mol/A^2 -> kJ/mol/nm^2
_KCAL_A2_TO_KJ_NM2 = 418.4


def available_platforms() -> set[str]:
    return {openmm.Platform.getPlatform(i).getName()
            for i in range(openmm.Platform.getNumPlatforms())}


def _platform(cfg):
    """Pick the best available platform: configured -> CUDA -> OpenCL -> CPU.

    On Colab the pip OpenMM wheel often exposes **OpenCL** (a genuine GPU
    platform on the NVIDIA T4) but not CUDA, so falling back to OpenCL keeps the
    run on the GPU instead of silently dropping to CPU.
    """
    avail = available_platforms()
    for name in (cfg.platform, "CUDA", "OpenCL", "CPU", "Reference"):
        if name in avail:
            props = {"Precision": cfg.precision} if name in ("CUDA", "OpenCL") else {}
            return openmm.Platform.getPlatformByName(name), props
    return openmm.Platform.getPlatformByName("Reference"), {}


def _integrator(cfg):
    integ = openmm.LangevinMiddleIntegrator(
        cfg.temperature_K * unit.kelvin,
        cfg.friction_ps / unit.picosecond,
        cfg.timestep_fs * unit.femtoseconds,
    )
    integ.setRandomNumberSeed(int(cfg.seed))
    return integ


def _constraints(cfg):
    return {"HBonds": app.HBonds, "AllBonds": app.AllBonds, "None": None}[cfg.constraints]


def build_system(prmtop_path: Path, cfg):
    """Load an AMBER prmtop and create a PME system (HMR-aware)."""
    prmtop = app.AmberPrmtopFile(str(prmtop_path))
    kw = dict(
        nonbondedMethod=app.PME,
        nonbondedCutoff=cfg.nonbonded_cutoff_nm * unit.nanometer,
        constraints=_constraints(cfg),
        rigidWater=True,
    )
    if cfg.hmr:
        kw["hydrogenMass"] = cfg.hydrogen_mass_amu * unit.amu
    system = prmtop.createSystem(**kw)
    return prmtop, system


def _solute_heavy_atoms(topology) -> list[int]:
    solvent = {"HOH", "WAT", "NA", "CL", "Na+", "Cl-", "K+"}
    idx = []
    for atom in topology.atoms():
        if atom.residue.name in solvent:
            continue
        if atom.element is not None and atom.element.symbol == "H":
            continue
        idx.append(atom.index)
    return idx


def _add_position_restraint(system, positions, atom_indices, k_kcal):
    force = openmm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    force.addGlobalParameter("k", k_kcal * _KCAL_A2_TO_KJ_NM2)
    for p in ("x0", "y0", "z0"):
        force.addPerParticleParameter(p)
    pos = positions.value_in_unit(unit.nanometer)
    for i in atom_indices:
        force.addParticle(i, [pos[i][0], pos[i][1], pos[i][2]])
    system.addForce(force)
    return force


@dataclass
class StageResult:
    output: Path
    ok: bool
    seconds: float = 0.0
    info: dict = None


def minimize(prmtop_path, rst7_path, out_dir: Path, cfg) -> StageResult:
    out = Path(out_dir) / "minimized_state.xml"
    if out.exists():
        return StageResult(out, ok=True, info={"reused": True})
    t0 = time.perf_counter()
    prmtop, system = build_system(prmtop_path, cfg)
    inpcrd = app.AmberInpcrdFile(str(rst7_path))
    integ = _integrator(cfg)
    plat, props = _platform(cfg)
    sim = app.Simulation(prmtop.topology, system, integ, plat, props)
    sim.context.setPositions(inpcrd.positions)
    if inpcrd.boxVectors is not None:
        sim.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
    sim.minimizeEnergy(maxIterations=cfg.minimize_max_iter)
    state = sim.context.getState(getPositions=True, getEnergy=True, getVelocities=True)
    with out.open("w") as fh:
        fh.write(openmm.XmlSerializer.serialize(state))
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    return StageResult(out, ok=True, seconds=round(time.perf_counter() - t0, 1),
                       info={"potential_energy_kJ_mol": energy, "platform": plat.getName()})


def equilibrate(prmtop_path, min_state_xml, out_dir: Path, cfg) -> StageResult:
    out = Path(out_dir) / "equilibrated.chk"
    if out.exists():
        return StageResult(out, ok=True, info={"reused": True})
    t0 = time.perf_counter()
    prmtop, system = build_system(prmtop_path, cfg)

    with open(min_state_xml) as fh:
        min_state = openmm.XmlSerializer.deserialize(fh.read())
    positions = min_state.getPositions()

    schedule = list(cfg.restraint_schedule) or [0.0]
    heavy = _solute_heavy_atoms(prmtop.topology)
    restraint = _add_position_restraint(system, positions, heavy, schedule[0])
    system.addForce(openmm.MonteCarloBarostat(
        cfg.pressure_bar * unit.bar, cfg.temperature_K * unit.kelvin, 25))

    integ = _integrator(cfg)
    plat, props = _platform(cfg)
    sim = app.Simulation(prmtop.topology, system, integ, plat, props)
    sim.context.setPositions(positions)
    if min_state.getPeriodicBoxVectors() is not None:
        sim.context.setPeriodicBoxVectors(*min_state.getPeriodicBoxVectors())
    sim.context.setVelocitiesToTemperature(cfg.temperature_K * unit.kelvin, int(cfg.seed))

    # NVT heat (barostat is present but restraint holds solute); then NPT with
    # the restraint stepped down across the schedule.
    nvt_steps = cfg.n_steps(cfg.nvt_ns)
    sim.step(nvt_steps)
    npt_steps_per_stage = max(1, cfg.n_steps(cfg.npt_ns) // len(schedule))
    for k in schedule:
        sim.context.setParameter("k", k * _KCAL_A2_TO_KJ_NM2)
        sim.step(npt_steps_per_stage)

    sim.saveCheckpoint(str(out))
    return StageResult(out, ok=True, seconds=round(time.perf_counter() - t0, 1),
                       info={"nvt_steps": nvt_steps, "restraint_schedule": schedule})


def produce(prmtop_path, equil_chk, out_dir: Path, cfg) -> StageResult:
    out_dir = Path(out_dir)
    traj = out_dir / "trajectory.dcd"
    chk = out_dir / "production.chk"
    csv = out_dir / "statedata.csv"
    progress_path = out_dir / "production_progress.json"
    target = cfg.n_steps(cfg.production_ns)
    t0 = time.perf_counter()

    prmtop, system = build_system(prmtop_path, cfg)
    system.addForce(openmm.MonteCarloBarostat(
        cfg.pressure_bar * unit.bar, cfg.temperature_K * unit.kelvin, 25))
    integ = _integrator(cfg)
    plat, props = _platform(cfg)
    sim = app.Simulation(prmtop.topology, system, integ, plat, props)

    resuming = chk.exists() and progress_path.exists()
    if resuming:
        sim.loadCheckpoint(str(chk))
        done = json.loads(progress_path.read_text()).get("steps_done", 0)
    else:
        sim.loadCheckpoint(str(equil_chk))
        done = 0

    remaining = max(0, target - done)
    if remaining == 0:
        return StageResult(traj, ok=traj.exists(), info={"already_complete": True,
                                                         "steps_done": done})

    save_every = cfg.save_interval_steps()
    ckpt_every = cfg.checkpoint_interval_steps()
    sim.reporters.append(app.DCDReporter(str(traj), save_every, append=resuming))
    sim.reporters.append(app.StateDataReporter(
        str(csv), save_every, step=True, time=True, potentialEnergy=True,
        temperature=True, density=True, append=resuming))
    sim.reporters.append(app.CheckpointReporter(str(chk), ckpt_every))

    # run in checkpoint-sized chunks, persisting progress so a mid-run
    # disconnect resumes from the last chunk boundary.
    while done < target:
        chunk = min(ckpt_every, target - done)
        sim.step(chunk)
        done += chunk
        sim.saveCheckpoint(str(chk))
        progress_path.write_text(json.dumps({"steps_done": done, "target": target}))

    return StageResult(traj, ok=traj.exists(), seconds=round(time.perf_counter() - t0, 1),
                       info={"steps_done": done, "n_frames_est": done // save_every,
                             "platform": plat.getName()})
