import torch
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, "../../")
sys.path.insert(0, "./")
from mlcg.geometry.topology import Topology
from mlcg.geometry.topology import (
    get_connectivity_matrix,
    get_n_paths,
    get_improper_paths,
)
from mlcg.geometry.statistics import compute_statistics
from mlcg.geometry.statistics import _symmetrise_map
from mlcg.nn.prior import HarmonicBonds, HarmonicAngles, Repulsion, Dihedral, Coulombic
from mlcg.data.atomic_data import AtomicData
from mlcg.neighbor_list.neighbor_list import make_neighbor_list
import mdtraj as md

from torch_geometric.data.collate import collate
from ase.build import molecule

# Physical units
temperature = 350  # K
#:Boltzmann constant in kcal/mol/K
kB = 0.0019872041
beta = 1 / (temperature * kB)

# Make a few pertubed frames of trans-butane using gaussian noise
mol = molecule("trans-butane")
ref_coords = np.array(mol.get_positions())

mock_data_frames = []
from numpy.random import default_rng

rng = default_rng(94834)
for i in range(1000):
    perturbed_coords = ref_coords + 0.4 * rng.standard_normal(ref_coords.shape)
    mock_data_frames.append(torch.tensor(perturbed_coords))
mock_data_frames = torch.stack(mock_data_frames, dim=0)

# Topology object for the above molecule
test_topo = Topology.from_ase(mol)
conn_mat = get_connectivity_matrix(test_topo)
dihedral_paths = get_n_paths(conn_mat, n=4)
improper_paths = get_improper_paths(conn_mat, n=4)
test_topo.dihedrals_from_edge_index(dihedral_paths)
test_topo.impropers_from_edge_index(improper_paths)

# unique bond/angle species

bond_edges = test_topo.bonds2torch()
angle_edges = test_topo.angles2torch()
dihedral_edges = test_topo.dihedrals2torch()
improper_edges = test_topo.impropers2torch()
print(bond_edges)
# Add some nonbonded edges
non_bonded_edges = torch.tensor(
    [
        [0, 0, 0, 1, 2, 3, 3],
        [2, 3, 9, 5, 6, 10, 10],
    ]
)

bond_species = torch.tensor(test_topo.types)[bond_edges]
angle_species = torch.tensor(test_topo.types)[angle_edges]
non_bond_species = torch.tensor(test_topo.types)[non_bonded_edges]
dihedral_species = torch.tensor(test_topo.types)[dihedral_edges]
improper_speces = torch.tensor(test_topo.types)[improper_edges]

nls_tags = ["bonds", "angles", "non-bonded", "dihedrals", "impropers","coulombic"]
nls_orders = [2, 3, 2, 4, 2, 2]
nls_edges = [
    bond_edges,
    angle_edges,
    non_bonded_edges,
    dihedral_edges,
    improper_edges,
    non_bonded_edges,
]
cell_shifts = [
    None,
    None,
    None,
    None,
    None,
    cell_shift,
]
data_list = []

for frame in range(mock_data_frames.shape[0]):
    neighbor_lists = {}
    for (tag, order, edge_list) in zip(nls_tags, nls_orders, nls_edges):
        neighbor_lists[tag] = make_neighbor_list(tag, order, edge_list)
        data_point = AtomicData(
            pos=mock_data_frames[frame],
            atom_types=torch.tensor(test_topo.types),
            neighbor_list=neighbor_lists,
        )
        data_list.append(data_point)

collated_data, _, _ = collate(
    data_list[0].__class__, data_list=data_list, increment=True, add_batch=True
)


def test_unique_species(
    test_data, target, beta, target_prior, expected_species
):
    """Tests to make sure the correct feature species survive statistics gathering"""
    statistics = compute_statistics(test_data, target, beta, target_prior)
    order = test_data.neighbor_list[target]["index_mapping"].shape[0]
    species_groups = torch.tensor(list(statistics.keys())).t()

    # reduce to symmetrised and unique group tuples for comparison
    unique_expected_groups = torch.unique(
        _symmetrise_map[order](expected_species), dim=1
    ).numpy()

    unique_species_groups = torch.unique(
        _symmetrise_map[order](species_groups), dim=1
    ).numpy()
    unique_species_groups = [
        tuple(sorted([*group])) for group in unique_species_groups.T
    ]
    unique_expected_groups = [
        tuple(sorted([*group])) for group in unique_expected_groups.T
    ]

    assert len(unique_species_groups) == len(unique_expected_groups)
    for group in unique_species_groups:
        assert group in unique_expected_groups


def test_histogram_options(
    test_data, target, beta, target_prior, nbins, b_min, b_max
):
    """Test to make sure histogram bin/range options are respected
    for various end options
    """
    statistics = compute_statistics(
        test_data, target, beta, target_prior, nbins, b_min, b_max
    )

    for species_group in statistics.keys():
        p = statistics[species_group]["p"].numpy()
        p_bin = statistics[species_group]["p_bin"].numpy()
        assert len(p) == nbins
        assert len(p_bin) == nbins
        # case if lower bound is specified
        if b_min != None:
            delta = p_bin[1] - p_bin[0]
            assert p_bin[0] == pytest.approx(b_min + 0.5 * delta, 6)
        # case if upper bound is specified
        if b_max != None:
            delta = p_bin[1] - p_bin[0]
            assert p_bin[-1] == pytest.approx(b_max - 0.5 * delta, 6)
        # case if both bounds are specified
        if b_min != None and b_max != None:
            bins = np.linspace(b_min, b_max, nbins + 1)
            delta = bins[1] - bins[0]
            assert p_bin[0] == (b_min + 0.5 * delta)
            assert p_bin[-1] == (b_max - 0.5 * delta)


def test_torsion_options(
    test_data, traj
):
    """
        Test that torsions are being computed
    """

    target = "dihedrals"
    target_prior = Dihedral

    _,phi = md.compute_phi(traj)
    mapping = test_data.neighbor_list[target]["index_mapping"]
    own_phi = target_prior.compute_features(test_data.pos, mapping)

    for ix in range(traj.n_frames):
        assert np.abs(phi[ix]-own_phi[ix].numpy()) < 1e-5

