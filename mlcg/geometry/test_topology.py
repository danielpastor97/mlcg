import networkx as nx
from mlcg.geometry.topology import *
import torch
import pytest
import numpy as np

# make a simple molecule defined by the following bonded
# topology

atom_types = [1, 6, 2, 5, 4, 9, 8, 2, 6, 4, 7]
atom_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
bond_edges = torch.tensor(
    [[0, 1, 1, 3, 4, 3, 6, 6, 8, 8], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
)

cmat_undirected = torch.tensor(
    [
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    ]
)

cmat_directed = torch.tensor(
    [
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)

test_topo = Topology()

for atom_type, name in zip(atom_types, atom_names):
    test_topo.add_atom(atom_type, name)

test_topo.bonds_from_edge_index(bond_edges)


@pytest.mark.parametrize(
    "topo_input,cmat_expected,directed",
    [(test_topo, cmat_undirected, False), (test_topo, cmat_directed, True)],
)
def test_connectivity_matrix(topo_input, cmat_expected, directed):
    constructed_cmat = get_connectivity_matrix(topo_input, directed).numpy()
    np.testing.assert_array_equal(constructed_cmat, cmat_expected.numpy())
