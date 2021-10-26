import networkx as nx
from mlcg.geometry.topology import *
from mlcg.geometry.statistics import *
from mlcg.geometry.statistics import _symmetrise_map
from mlcg.nn.prior import *
from mlcg.data import *
from torch_geometric.data.collate import collate
import torch
import pytest
import numpy as np

# Physical units
temperature = 350  # K
#:Boltzmann constan in kcal/mol/K
kB = 0.0019872041
beta = 1 / (temperature * kB)

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

# unique bond/angle species
bond_species = torch.tensor(test_topo.types)[bond_edges]
angle_species = torch.tensor(test_topo.types)[bonded_angles]
non_bond_species = torch.tensor(test_topo.types)[edges_1_5]

# Mock data - temporarily its random
n_frames = 10
rand_coords = torch.randn(10, 11, 3)
nls_tags = ["bonds", "angles", "non-bonded"]
nls_orders = [2, 3, 2]
nls_edges = [bond_edges, bonded_angles, edges_1_5]
data_list = []

for frame in range(rand_coords.shape[0]):
    neighbor_lists = {}
    for (tag, order, edge_list) in zip(nls_tags, nls_orders, nls_edges):
        neighbor_lists[tag] = {
            "tag": tag,
            "order": order,
            "index_mapping": edge_list,
            "cell_shifts": None,
            "rcut": None,
            "self_interaction": False,
        }
        data_point = AtomicData(
            pos=rand_coords[frame],
            atom_types=torch.tensor(atom_types),
            neighbor_list=neighbor_lists,
        )
        data_list.append(data_point)

collated_data, _, _ = collate(
    data_list[0].__class__, data_list=data_list, increment=True, add_batch=True
)


@pytest.mark.parametrize(
    "test_data, target, beta, target_prior, expected_species",
    [
        (collated_data, "bonds", beta, HarmonicBonds, bond_species),
        (collated_data, "angles", beta, HarmonicAngles, angle_species),
        (collated_data, "non-bonded", beta, Repulsion, non_bond_species),
    ],
)
def test_unique_species(
    test_data, target, beta, target_prior, expected_species
):
    """Tests to make sure the correct feature species survive statistics gathering"""
    statistics = compute_statistics(test_data, target, beta, target_prior)
    order = test_data.neighbor_list[target]["index_mapping"].shape[0]
    species_groups = torch.tensor(list(statistics.keys())).t()
    unique_expected_groups = torch.unique(
        _symmetrise_map[order](expected_species), dim=1
    ).numpy()

    # reduce to symmetrised and unique group tuples for comparison
    unique_species_groups = torch.unique(
        _symmetrise_map[order](species_groups), dim=1
    ).numpy()
    unique_species_groups = [
        tuple(sorted([*group])) for group in unique_species_groups.T
    ]
    unique_expected_groups = [
        tuple(sorted([*group])) for group in unique_expected_groups.T
    ]
    print(sorted(unique_species_groups))
    print(sorted(unique_expected_groups))
    assert len(unique_species_groups) == len(unique_expected_groups)
    for group in unique_species_groups:
        assert group in unique_expected_groups
