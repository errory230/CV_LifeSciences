"""Step: SMILES -> a single low-energy 3D conformer (RDKit, CPU).

We embed several conformers with ETKDGv3 and minimize each with MMFF94s, then
keep the lowest-energy one as the starting geometry for parameterization. This
is deliberately just a *starting* structure: the real conformational ensemble is
produced later by solvent MD. A modest multi-conformer search here only avoids
handing the parameterizer a strained or unrepresentative geometry.

Conformer generation is seeded (`random_seed`) so the whole prep is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import AllChem


@dataclass
class ConformerResult:
    mol: Chem.Mol  # embedded, H-explicit, single best conformer
    energy: float  # MMFF energy of the kept conformer (kcal/mol)
    n_conformers_tried: int
    n_conformers_converged: int
    forcefield: str


def generate_conformer(
    smiles: str,
    *,
    random_seed: int = 42,
    n_conformers: int = 20,
    max_iters: int = 400,
) -> ConformerResult:
    """Embed `n_conformers` ETKDGv3 conformers, MMFF-minimize, return the best."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.useRandomCoords = True
    cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params))
    if not cids:
        # Fallback: a single embed with random coords for awkward macrocycles.
        cid = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=random_seed)
        if cid < 0:
            raise RuntimeError(f"conformer embedding failed for {smiles!r}")
        cids = [cid]

    # MMFF94s minimization of every embedded conformer.
    ff_name = "MMFF94s"
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=ff_name)
    energies: list[tuple[float, int]] = []
    n_converged = 0
    for cid in cids:
        if props is None:
            # No MMFF params (rare heteroatoms): fall back to UFF.
            ff_name = "UFF"
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
        else:
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
        converged = ff.Minimize(maxIts=max_iters)
        if converged == 0:
            n_converged += 1
        energies.append((ff.CalcEnergy(), cid))

    energies.sort(key=lambda t: t[0])
    best_energy, best_cid = energies[0]

    # Keep only the best conformer on the returned mol.
    best = Chem.Mol(mol)
    best.RemoveAllConformers()
    best.AddConformer(mol.GetConformer(best_cid), assignId=True)

    return ConformerResult(
        mol=best,
        energy=round(best_energy, 4),
        n_conformers_tried=len(cids),
        n_conformers_converged=n_converged,
        forcefield=ff_name,
    )


def write_sdf(mol: Chem.Mol, path: str) -> None:
    """Write the 3D H-explicit mol to an SDF (input for antechamber)."""
    writer = Chem.SDWriter(path)
    writer.write(mol)
    writer.close()
