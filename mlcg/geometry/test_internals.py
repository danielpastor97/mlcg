import torch
import numpy as np
import mdtraj
from mdtraj.core.element import carbon
from mlcg.geometry.internal_coordinates import *

# Mock data of ten atoms for 1000 frames
n_frames = 1000
n_atoms = 10
test_coords = np.random.randn(n_frames, n_atoms, 3).astype("float32")
torch_coords = torch.tensor(test_coords.reshape(n_frames * n_atoms, 3))

dihedral_atoms = np.array(
    [[i, i + 1, i + 2, i + 3] for i in range(test_coords.shape[1] - 3)]
)
mapping = torch.tensor(
    np.concatenate(
        [dihedral_atoms.T + (10 * i) for i in range(n_frames)], axis=1
    )
)

topo = mdtraj.core.topology.Topology()
topo.add_chain()
topo.add_residue("ALA1", topo._chains[0])
for _ in range(n_atoms):
    topo.add_atom("CA", carbon, list(topo.residues)[0])
traj = mdtraj.core.trajectory.Trajectory(test_coords, topo)


def test_mdtraj_dihedral_compatibility():
    """Tests to make sure that MLCG torsion calculations
    aggree with the dihedral calculations from mlcg
    """

    mlcg_dihedrals = (
        compute_torsions(torch_coords, mapping)
        .numpy()
        .reshape(n_frames, dihedral_atoms.shape[0])
    )
    mdtraj_dihedrals = mdtraj.compute_dihedrals(traj, dihedral_atoms)
    np.testing.assert_allclose(mlcg_dihedrals, mdtraj_dihedrals, rtol=1e-3)
