import networkx as nx
from mlcg.geometry.statistics import *
from mlcg.nn.prior import *
from torch_geometric.data.collate import collate
import torch
import pytest
import numpy as np

# make a simple molecule defined by the following bonded
# topology, with dummy atom types and names.

atom_types = [1, 6, 2, 5, 4, 9, 8, 2, 6, 4, 7]
atom_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
bond_edges = torch.tensor(
    [[0, 1, 1, 3, 4, 3, 6, 6, 8, 8], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
)

# All of the 1-5 (unique) distance pairs
edges_1_5 = torch.tensor(
    [
        [0, 0, 0, 1, 1, 2, 2, 2, 4, 4, 5, 5],
        [5, 7, 8, 9, 10, 5, 7, 8, 9, 10, 7, 8],
    ]
)

# All of the (unique) 1-3 bonded angles
bonded_angles = torch.tensor(
    [
        [0, 0, 1, 1, 2, 3, 3, 3, 4, 6, 6, 7, 9],
        [1, 1, 3, 3, 1, 4, 6, 6, 3, 8, 8, 6, 8],
        [2, 3, 4, 6, 3, 5, 7, 8, 6, 9, 10, 8, 10],
    ]
)

# Topology object for the above molecule
test_topo = Topology()

# Adding atoms
for atom_type, name in zip(atom_types, atom_names):
    test_topo.add_atom(atom_type, name)

# Adding bonds
test_topo.bonds_from_edge_index(bond_edges)


# We should test the pipeline for each prior




