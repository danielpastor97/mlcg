from typing import List, Callable
import warnings
import torch
import pytest
import numpy as np
from numpy.random import default_rng
from torch_geometric.data.collate import collate

from ase.build import molecule
from mlcg.geometry import Topology
from mlcg.geometry.statistics import *
from mlcg.neighbor_list.neighbor_list import make_neighbor_list
from mlcg.simulation import *
from mlcg.nn import *
from mlcg.data._keys import *

# Seeding
rng = default_rng(94834)

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
priors = {
    name: GradientsOut(priors[name], targets=[FORCE_KEY])
    for name in priors.keys()
}
full_model = SumOut(priors)

# 5 replicas starting from the same structure
# For both Langevin (massive) and Overdamped (massless) cases
massless_initial_data_list = []
initial_data_list = []
for frame in range(5):
    neighbor_lists = {}
    for (tag, order, edge_list) in zip(nls_tags, nls_orders, nls_edges):
        neighbor_lists[tag] = make_neighbor_list(tag, order, edge_list)
    data_point = AtomicData(
        pos=torch.tensor(mol.get_positions()),
        atom_types=torch.tensor(test_topo.types),
        masses=torch.tensor(mol.get_masses()),
        cell=None,
        velocities=None,
        neighbor_list=neighbor_lists,
    )
    massless_data_point = AtomicData(
        pos=torch.tensor(mol.get_positions()),
        atom_types=torch.tensor(test_topo.types),
        cell=None,
        velocities=None,
        neighbor_list=neighbor_lists,
    )

    initial_data_list.append(data_point)
    massless_initial_data_list.append(massless_data_point)

### ================================================== ###
### Input data lists designed to raise errors/warnings ###
### ================================================== ###


def generate_broken_data_list(
    key: str, corruptor: Callable
) -> List[AtomicData]:
    """Helper function to generate broken data lists

    Parameters
    ----------
    key:
        key of the data list that should be corrupted
    corruptor:
        Anonynous (lambda) function that takes the current
        frame of the data list and conditionally returns
        different values

    Returns
    -------
    broken_data_list:
        List of AtomicData instances that has been corrupted
        at the frame and with the damage specified by the
        the corruptor
    """

    broken_data_list = []
    for frame in range(5):
        neighbor_lists = {}
        for (tag, order, edge_list) in zip(nls_tags, nls_orders, nls_edges):
            neighbor_lists[tag] = make_neighbor_list(tag, order, edge_list)
        data_point = AtomicData(
            pos=torch.tensor(mol.get_positions()),
            atom_types=torch.tensor(test_topo.types),
            masses=torch.tensor(mol.get_masses()),
            cell=None,
            velocities=None,
            neighbor_list=neighbor_lists,
        )
        broken_data_list.append(data_point)
    # corrupt a frame
    for frame in range(5):
        data_point[key] = corruptor(frame)
    return broken_data_list


### corruptors - lambdas that introduce a problem in the data list ###

# Puts the wrong mass on the fourth frame
wrong_mass_fn = (
    lambda x: 2 * torch.tensor(mol.get_masses())
    if x == 3
    else torch.tensor(mol.get_masses())
)

# Gives a structure with the wrong shape on the third frame
wrong_struct_fn = (
    lambda x: torch.randn(7, 3) if x == 2 else torch.tensor(mol.get_positions())
)
# Gives the wrong atomic types on the second frame
wrong_atom_fn = (
    lambda x: 7 * torch.tensor(mol.get_atomic_numbers())
    if x == 1
    else torch.tensor(mol.get_atomic_numbers())
)

wrong_mass_data_list = generate_broken_data_list(MASS_KEY, wrong_mass_fn)
wrong_struct_data_list = generate_broken_data_list(
    POSITIONS_KEY, wrong_struct_fn
)
wrong_atom_data_list = generate_broken_data_list(ATOM_TYPE_KEY, wrong_atom_fn)


@pytest.mark.parametrize(
    "full_model, initial_data_list, sim_kwargs, expected_raise",
    [
        (
            # Should raise error: one frame has different masses
            full_model,
            wrong_mass_data_list,
            {},
            ValueError,
        ),
        (
            # Should raise error: one frame has a different structure
            full_model,
            wrong_struct_data_list,
            {},
            ValueError,
        ),
        (
            # Should raise error: one frame has a different atom types
            full_model,
            wrong_atom_data_list,
            {},
            ValueError,
        ),
    ],
)
def test_data_list__raises(
    full_model, initial_data_list, sim_kwargs, expected_raise
):
    """Test to make sure certain warnings/errors are raised regarding the data list"""
    if isinstance(expected_raise, Exception):
        with pytest.raises(expected_raise):
            sim = _Simulation(full_model, initial_data_list, **sim_kwargs)
    if isinstance(expected_raise, UserWarning):
        with pytest.warns(expected_raise):
            sim = _Simulation(full_model, initial_data_list, **sim_kwargs)


@pytest.mark.parametrize(
    "full_model, initial_data_list, sim_class, sim_args, sim_kwargs",
    [
        (full_model, massless_initial_data_list, OverdampedSimulation, [], {}),
        (
            full_model,
            initial_data_list,
            LangevinSimulation,
            [],
            {"friction": 1.0},
        ),
    ],
)
def test_simulation_run(
    full_model, initial_data_list, sim_class, sim_args, sim_kwargs
):
    """Test to make sure the simulation runs"""
    simulation = sim_class(
        full_model, initial_data_list, *sim_args, **sim_kwargs
    )
    simulation.simulate()
