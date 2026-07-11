"""Per-frame 3D descriptors recomputed from a solute trajectory (mdtraj).

The analysis stage already stores 3D-PSA and RMSD per frame; here we add the
other ensemble descriptors the Stage-3 tooling needs — radius of gyration,
total SASA, and intramolecular hydrogen-bond count — with explicit, recorded
criteria. Intramolecular H-bonds and a compact (low-Rg / high-IMHB) shape are
the structural signature of the "chameleon" behaviour that governs passive
permeability, so they sit next to 3D-PSA as the core ensemble descriptors.

All functions take an already solvent-stripped `mdtraj.Trajectory`.
"""

from __future__ import annotations

from dataclasses import dataclass

import mdtraj as md
import numpy as np


@dataclass
class HBondCriteria:
    donor_acceptor_A: float = 3.5   # max donor-heavy ... acceptor distance
    angle_deg: float = 120.0        # min D-H-A angle


def radius_of_gyration(traj) -> np.ndarray:
    return md.compute_rg(traj) * 10.0  # nm -> A


def total_sasa(traj, probe_A: float = 1.4) -> np.ndarray:
    sasa = md.shrake_rupley(traj, mode="atom", probe_radius=probe_A / 10.0)
    return sasa.sum(axis=1) * 100.0  # nm^2 -> A^2


def _hbond_triplets(top) -> list[tuple[int, int, int]]:
    """(donor_heavy, H, acceptor) candidate triplets for intramolecular HBs."""
    # map each hydrogen to the heavy atom it is bonded to
    h_to_donor: dict[int, int] = {}
    bonded: set[tuple[int, int]] = set()
    for a, b in top.bonds:
        bonded.add((a.index, b.index)); bonded.add((b.index, a.index))
        for h, heavy in ((a, b), (b, a)):
            if h.element and h.element.symbol == "H" and heavy.element \
                    and heavy.element.symbol in ("N", "O"):
                h_to_donor[h.index] = heavy.index
    acceptors = [at.index for at in top.atoms
                 if at.element and at.element.symbol in ("N", "O")]
    triplets = []
    for h, donor in h_to_donor.items():
        for acc in acceptors:
            if acc == donor or (donor, acc) in bonded:
                continue
            triplets.append((donor, h, acc))
    return triplets


def intramolecular_hbonds(traj, crit: HBondCriteria | None = None) -> np.ndarray:
    """Per-frame count of intramolecular H-bonds (distance + angle criteria)."""
    crit = crit or HBondCriteria()
    triplets = _hbond_triplets(traj.topology)
    n = traj.n_frames
    if not triplets:
        return np.zeros(n, dtype=int)
    da_pairs = np.array([[d, a] for d, _, a in triplets])
    dha = np.array(triplets)
    dist = md.compute_distances(traj, da_pairs) * 10.0            # (n, m) in A
    angle = np.degrees(md.compute_angles(traj, dha))              # (n, m) in deg
    ok = (dist <= crit.donor_acceptor_A) & (angle >= crit.angle_deg)
    return ok.sum(axis=1).astype(int)


def frame_descriptors(traj, *, probe_A: float = 1.4,
                      hbond: HBondCriteria | None = None) -> dict[str, np.ndarray]:
    """Compute rg / sasa / intra_hbond for every frame of a solute trajectory."""
    return {
        "rg": radius_of_gyration(traj),
        "sasa": total_sasa(traj, probe_A=probe_A),
        "intra_hbond": intramolecular_hbonds(traj, hbond),
    }
