import torch
import pytest
import numpy as np
from torch_geometric.data.collate import collate

from ase.build import molecule
from mlcg.geometry import Topology
from mlcg.geometry.statistics import fit_baseline_models
from mlcg.neighbor_list.neighbor_list import make_neighbor_list
from mlcg.nn.prior import HarmonicBonds, HarmonicAngles
from mlcg.nn.gradients import SumOut, GradientsOut
from mlcg.data._keys import ENERGY_KEY, FORCE_KEY
from mlcg.data.atomic_data import AtomicData

# Seeding
rng = np.random.default_rng(94834)

# Physical units
temperature = 350  # K
#:Boltzmann constan in kcal/mol/K
kB = 0.0019872041
beta = 1 / (temperature * kB)

# Here we make a simple prior-only model of aluminum-fluoride
mol = molecule("AlF3")
test_topo = Topology.from_ase(mol)
n_atoms = len(test_topo.types)
initial_coords = np.array(mol.get_positions())

prior_data_frames = []
for i in range(1000):
    perturbed_coords = initial_coords + 0.2 * rng.standard_normal(
        initial_coords.shape
    )
    prior_data_frames.append(torch.tensor(perturbed_coords))
prior_data_frames = torch.stack(prior_data_frames, dim=0)

# Set up some data with bond/angle neighborlists:
bond_edges = test_topo.bonds2torch()
angle_edges = test_topo.angles2torch()

# Generete some noisy data for the priors
nls_tags = ["bonds", "angles"]
nls_orders = [2, 3]
nls_edges = [bond_edges, angle_edges]

prior_data_list = []
for frame in range(prior_data_frames.shape[0]):
    neighbor_lists = {}
    for (tag, order, edge_list) in zip(nls_tags, nls_orders, nls_edges):
        neighbor_lists[tag] = make_neighbor_list(tag, order, edge_list)
    data_point = AtomicData(
        pos=prior_data_frames[frame],
        atom_types=torch.tensor(test_topo.types),
        masses=torch.tensor(mol.get_masses()),
        cell=None,
        neighbor_list=neighbor_lists,
    )
    prior_data_list.append(data_point)

collated_prior_data, _, _ = collate(
    prior_data_list[0].__class__,
    data_list=prior_data_list,
    increment=True,
    add_batch=True,
)
# Fit the priors
prior_cls = [HarmonicBonds, HarmonicAngles]
priors, stats = fit_baseline_models(collated_prior_data, beta, prior_cls)

# Construct the model
priors = {
    name: GradientsOut(priors[name], targets=[FORCE_KEY])
    for name in priors.keys()
}
full_model = SumOut(priors)

atom_types = torch.tensor(mol.get_atomic_numbers())
force_shape = collated_prior_data.pos.shape
energy_shape = torch.Size([len(prior_data_list)])


@pytest.mark.parametrize(
    "model, collated_data, out_targets, expected_shapes",
    [
        (
            full_model,
            collated_prior_data,
            [ENERGY_KEY, FORCE_KEY],
            [energy_shape, force_shape],
        )
    ],
)
def test_outs(model, collated_data, out_targets, expected_shapes):
    """Test to make sure that the output dictionary is properly populated
    and that the correspdonding shapes of the outputs are correct given the
    requested gradient targets.
    """
    collated_data = model(collated_data)
    assert len(collated_data.out) != 0
    names = model.models.keys()
    for name in names:
        assert name in collated_data.out.keys()

    for target, shape in zip(model.targets, expected_shapes):
        assert target in collated_data.out.keys()
        assert shape == collated_data.out[target].shape
