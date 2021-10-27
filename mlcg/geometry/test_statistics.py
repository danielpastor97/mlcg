import networkx as nx
from mlcg.geometry.topology import *
from mlcg.geometry.statistics import *
from mlcg.geometry.statistics import _symmetrise_map
from mlcg.nn.prior import *
from mlcg.data import *
from mlcg.neighbor_list.utils import ase_bonds2tensor, ase_angles2tensor

from torch_geometric.data.collate import collate
from ase.build import molecule
from ase.geometry.analysis import Analysis
from ase.neighborlist import natural_cutoffs
import torch
import pytest
import numpy as np

# Physical units
temperature = 350  # K
#:Boltzmann constan in kcal/mol/K
kB = 0.0019872041
beta = 1 / (temperature * kB)

# Make a few pertubed frames of trans-butane using gaussian noise
mol = molecule("trans-butane")
analysis = Analysis(mol)  # , natural_cutoffs(mol))
bond_edges = ase_bonds2tensor(analysis)
angle_edges = ase_angles2tensor(analysis)
# Add some non-bonded edges
non_bonded_edges = torch.tensor(
    [[0, 0, 0, 1, 2, 3, 3], [2, 3, 9, 5, 6, 10, 10],]
)
ref_coords = np.array(mol.get_positions())

mock_data_frames = []
for i in range(1000):
    perturbed_coords = ref_coords + np.random.rand(*ref_coords.shape)
    mock_data_frames.append(torch.tensor(perturbed_coords))
mock_data_frames = torch.stack(mock_data_frames, dim=0)

# Topology object for the above molecule
test_topo = Topology()

# Adding atoms
for atom_type, name in zip(
    mol.get_atomic_numbers(), [str(num) for num in mol.get_atomic_numbers()]
):
    test_topo.add_atom(atom_type, name)

# Adding bonds/angles
test_topo.bonds_from_edge_index(bond_edges)
test_topo.angles_from_edge_index(angle_edges)

# unique bond/angle species
bond_species = torch.tensor(test_topo.types)[bond_edges]
angle_species = torch.tensor(test_topo.types)[angle_edges]
non_bond_species = torch.tensor(test_topo.types)[non_bonded_edges]

nls_tags = ["bonds", "angles", "non-bonded"]
nls_orders = [2, 3, 2]
nls_edges = [bond_edges, angle_edges, non_bonded_edges]
data_list = []

for frame in range(mock_data_frames.shape[0]):
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
            pos=mock_data_frames[frame],
            atom_types=torch.tensor(test_topo.types),
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


# Test to make sure histograms are okay


@pytest.mark.parametrize(
    "test_data, target, beta, target_prior, nbins, amin, amax",
    [
        (collated_data, "bonds", beta, HarmonicBonds, 10, None, None),
        (collated_data, "bonds", beta, HarmonicBonds, 10, -2, None),
        (collated_data, "bonds", beta, HarmonicBonds, 10, None, 2),
        (collated_data, "bonds", beta, HarmonicBonds, 10, -2, 2),
    ],
)
def test_histogram_options(
    test_data, target, beta, target_prior, nbins, amin, amax
):
    """Test to make sure histogram bin/range options are respected
    for various end options
    """
    statistics = compute_statistics(
        test_data, target, beta, target_prior, nbins, amin, amax
    )

    for species_group in statistics.keys():
        p = statistics[species_group]["p"].numpy()
        p_bin = statistics[species_group]["p_bin"].numpy()
        assert len(p) == nbins
        assert len(p_bin) == nbins
        # case if lower bound is specified
        if amin != None:
            delta = p_bin[1] - p_bin[0]
            assert p_bin[0] == pytest.approx(amin + 0.5 * delta, 6)
        # case if upper bound is specified
        if amax != None:
            delta = p_bin[1] - p_bin[0]
            assert p_bin[-1] == pytest.approx(amax - 0.5 * delta, 6)
        # case if both bounds are specified
        if amin != None and amax != None:
            bins = np.linspace(amin, amax, nbins + 1)
            delta = bins[1] - bins[0]
            assert p_bin[0] == (amin + 0.5 * delta)
            assert p_bin[-1] == (amax - 0.5 * delta)
