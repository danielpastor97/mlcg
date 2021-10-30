import networkx as nx
import torch
import pytest
import numpy as np
from torch_geometric.data.collate import collate

from ase.build import molecule
from mlcg.geometry import Topology
from mlcg.geometry.statistics import *
from mlcg.simulation import *
from mlcg.nn import *
from mlcg.data._keys import FORCE_KEY

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
    perturbed_coords = initial_coords + np.random.rand(*initial_coords.shape)
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
        neighbor_lists[tag] = {
            "tag": tag,
            "order": order,
            "index_mapping": edge_list,
            "cell_shifts": None,
            "rcut": None,
            "self_interaction": False,
        }
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
energy_model = nn.Sequential(*[priors[key] for key in priors.keys()])
energy_model.name = "Prior Model"
full_model = GradientsOut(energy_model, targets=FORCE_KEY)

# 5 replicas starting from the same structure
initial_coords = torch.stack(
    [torch.tensor(mol.get_positions()) for _ in range(5)]
)
atom_types = torch.tensor(mol.get_atomic_numbers())


@pytest.mark.parametrize(
    "full_model, initial_coords, atom_types, sim_kwargs",
    [(full_model, initial_coords, atom_types, {})],
)
def test_simulation_run(full_model, initial_coords, atom_types, sim_kwargs):
    """Test to make sure the simulation runs"""
    simulation = Simulation(
        full_model, initial_coords, atom_types, **sim_kwargs
    )
    simulation.simulate()
    pass
