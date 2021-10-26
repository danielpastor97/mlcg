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

edges_1_5 = torch.tensor(
    [
        [0, 0, 0, 1, 1, 2, 2, 2, 4, 4, 5, 5],
        [5, 7, 8, 9, 10, 5, 7, 8, 9, 10, 7, 8],
    ]
)

bonded_angles = torch.tensor(
    [
        [0, 0, 1, 1, 2, 3, 3, 3, 4, 6, 6, 7, 9],
        [1, 1, 3, 3, 1, 4, 6, 6, 3, 8, 8, 6, 8],
        [2, 3, 4, 6, 3, 5, 7, 8, 6, 9, 10, 8, 10],
    ]
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


@pytest.mark.parametrize("test_topo, pairs_expected", [(test_topo, edges_1_5)])
def test_n_pairs(test_topo, pairs_expected):
    cmat = get_connectivity_matrix(test_topo)
    recovered_pairs = get_n_pairs(cmat, n=5).numpy()
    np.testing.assert_array_equal(recovered_pairs, pairs_expected.numpy())


@pytest.mark.parametrize(
    "test_topo, edges_expected, n, symmetrise",
    [(test_topo, bonded_angles, 3, True),],
)
def test_n_paths(test_topo, edges_expected, n, symmetrise):
    cmat = get_connectivity_matrix(test_topo)
    recovered_paths = get_n_paths(cmat, n=n, symmetrise=symmetrise).numpy()
    np.testing.assert_array_equal(recovered_paths, edges_expected.numpy())
