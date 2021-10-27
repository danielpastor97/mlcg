import networkx as nx
import torch
from torch_geometric.data.collate import collate
import pytest
import numpy as np

from mlcg.nn import *
from mlcg.geometry import Topology
from mlcg.data.atomic_data import AtomicData
from mlcg.data._keys import ENERGY_KEY, FORCE_KEY
from ase.build import molecule

standard_basis = GaussianBasis()
standard_cutoff = IdentityCutoff(cutoff_lower=0, cutoff_upper=5)

# prepare some data with ASE
mol_names = [
    "AlF3",
    "C2H3",
    "ClF",
    "PF3",
    "PH2",
    "CH3CN",
    "cyclobutene",
    "CH3ONO",
    "SiH3",
    "C3H6_D3h",
    "CO2",
    "NO",
    "trans-butane",
    "H2CCHCl",
    "LiH",
    "NH2",
    "CH",
    "CH2OCH2",
    "C6H6",
    "CH3CONH2",
    "cyclobutane",
    "H2CCHCN",
    "butadiene",
    "C",
    "H2CO",
    "CH3COOH",
    "HCF3",
    "CH3S",
    "CS2",
]

test_molecules = [molecule(name) for name in mol_names]
test_topos = [Topology.from_ase(mol) for mol in test_molecules]

# each AtomicData is a simple, fully connected graph
data_list = []
for mol, topo in zip(test_molecules, test_topos):
    neighbor_list = topo.neighbor_list("fully connected")
    data = AtomicData(
        pos=torch.tensor(mol.get_positions()),
        atom_types=torch.tensor(mol.get_atomic_numbers()),
        neighbor_list=neighbor_list,
    )
    data_list.append(data)

# Collate data
collated_data, _, _ = collate(
    data_list[0].__class__,
    data_list=data_list,
    increment=True,
    add_batch=True,
)

force_shape = collated_data.pos.shape


@pytest.mark.parametrize(
    "basis, cutoff, expected_warning",
    [
        (GaussianBasis(0, 5), CosineCutoff(0, 5), None),
        (GaussianBasis(1, 5), CosineCutoff(0, 5), UserWarning),
    ],
)
def test_cutoff_warning(basis, cutoff, expected_warning):
    with pytest.warns(expected_warning):
        StandardSchNet(basis, cutoff, [128, 128])


def test_minimum_interaction_block():
    with pytest.raises(ValueError):
        StandardSchNet(
            standard_basis, standard_cutoff, [128, 128], num_interactions=-1
        )


@pytest.mark.parametrize(
    "collated_data, grad_targets, expected_shapes",
    [(collated_data, FORCE_KEY, [force_shape])],
)
def test_prediction(collated_data, grad_targets, expected_shapes):
    """Test to make sure that the output dictionary is properly populated
    and that the correspdonding shapes of the outputs are correct given the
    requested gradient targets.
    """
    test_schnet = StandardSchNet(standard_basis, standard_cutoff, [128, 128])
    model = GradientsOut(test_schnet, targets=grad_targets).double()
    collated_data = model(collated_data)
    assert len(collated_data.out) != 0
    assert "SchNet" in collated_data.out.keys()
    # assert scalar energy
    assert ENERGY_KEY in collated_data.out["SchNet"].keys()
    assert collated_data.out["SchNet"][ENERGY_KEY].shape == torch.Size(
        (collated_data.pos.shape[0], 1)
    )
    for target, shape in zip(model.targets, expected_shapes):
        assert target in collated_data.out["SchNet"].keys()
        assert collated_data.out["SchNet"][target].shape == shape
