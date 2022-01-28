import torch
import numpy as np
import mdtraj
from mlcg.geometry.internal_coordinates import *

# Mock data of ten atoms for 1000 frames
n_frames = 1000
n_atoms = 10
test_coords = 10.0* np.random.randn(n_frames,n_atoms,3)
torch_coords = torch.tensor(test_coords.reshape(n_frames * n_atoms, 3))

dihedral_atoms = np.array([[i,i+1,i+2,i+3] for i in range(test_coords.shape[1] - 3)])
mapping = torch.tensor(np.concatenate([dihedral_atoms.T + (10*i) for i in range(n_frames)], axis=1))
print(dihedral_atoms)
print(mapping.shape)

def test_mdtraj_dihedral_compatibility():
    """Tests to make sure that MLCG torsion calculations
    aggree with the dihedral calculations from mlcg
    """

    mlcg_dihedrals = compute_torsions(torch_coords, mapping).numpy()
    mdtraj_dihedrals = mdtraj.geometry.dihedral._dihedral(test_coords, dihedral_atoms)
    np.testing.array_equal(mlcg_dihedrals, mdtraj_dihedrals)
