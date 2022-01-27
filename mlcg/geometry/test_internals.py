import torch
import numpy as np
import mdtraj
from mlcg.geometry.internal_coordinates import *

# Mock data of ten atoms for 1000 frames
test_coords = 10.0* np.random.randn(1000,10,3)
torch_coords = torch.tensor(test_coords)

dihedral_beads = np.array([[i,i+1,i+2,i+2] for i in range(test_coords.shape[0] - 4)])
mapping = torch.tensor(dihedral_beads)

def test_mdtraj_dihedral_compatibility():
    """Tests to make sure that MLCG torsion calculations
    aggree with the dihedral calculations from mlcg
    """

    dihedral
