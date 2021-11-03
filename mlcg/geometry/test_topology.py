import torch
import pytest
import numpy as np

from mlcg.geometry.topology import (
    Topology,
    get_connectivity_matrix,
    get_n_paths,
    get_n_pairs,
)

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

# The undirected connectivity matrix associated with the bonded graph
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

# The directed connectivity matrix associated with the bonded graph
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

# Topology object for the above molecule
test_topo = Topology()

# Adding atoms
for atom_type, name in zip(atom_types, atom_names):
    test_topo.add_atom(atom_type, name)

# Adding bonds
test_topo.bonds_from_edge_index(bond_edges)


@pytest.mark.parametrize(
    "topo_input,cmat_expected,directed",
    [(test_topo, cmat_undirected, False), (test_topo, cmat_directed, True)],
)
def test_connectivity_matrix(topo_input, cmat_expected, directed):
    """Test to make sure that get_connectivity_matrix returns the
    expected connectivity matrix for directed and undirected cases
    """
    constructed_cmat = get_connectivity_matrix(topo_input, directed).numpy()
    np.testing.assert_array_equal(constructed_cmat, cmat_expected.numpy())


@pytest.mark.parametrize("test_topo, pairs_expected", [(test_topo, edges_1_5)])
def test_n_pairs(test_topo, pairs_expected):
    """Test to make sure that get_n_pairs returns the expected set of pairs"""
    cmat = get_connectivity_matrix(test_topo)
    recovered_pairs = get_n_pairs(cmat, n=5).numpy()
    np.testing.assert_array_equal(recovered_pairs, pairs_expected.numpy())


@pytest.mark.parametrize(
    "test_topo, edges_expected, n, unique",
    [
        (test_topo, bonded_angles, 3, True),
    ],
)
def test_n_paths(test_topo, edges_expected, n, unique):
    """Test to make sure that the 1-5 pairs are correctly returned by
    get_n_paths
    """
    cmat = get_connectivity_matrix(test_topo)
    recovered_paths = get_n_paths(cmat, n=n, unique=unique).numpy()
    np.testing.assert_array_equal(recovered_paths, edges_expected.numpy())
