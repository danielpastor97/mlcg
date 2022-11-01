import sys

sys.path.insert(0, "/net/scratch-sheldon/clarkt/tmp/mlcg-tools")

from typing import List, Callable, Dict
import tempfile
import torch
import numpy as np
import pytest
import unittest
from copy import deepcopy
from torch_geometric.data.collate import collate
from ase.atoms import Atoms
from ase.build import molecule

from mlcg.geometry import Topology
from mlcg.geometry.topology import get_connectivity_matrix, get_n_paths
from mlcg.geometry.statistics import fit_baseline_models
from mlcg.neighbor_list.neighbor_list import make_neighbor_list
from mlcg.nn.prior import HarmonicBonds, HarmonicAngles, Dihedral, Repulsion
from mlcg.nn.gradients import SumOut, GradientsOut
from mlcg.nn.schnet import StandardSchNet
from mlcg.nn.cutoff import CosineCutoff
from mlcg.nn.radial_basis import GaussianBasis
from mlcg.data._keys import ENERGY_KEY, FORCE_KEY
from mlcg.data.atomic_data import AtomicData
from mlcg.simulation.base import _Simulation
from mlcg.simulation.langevin import (
    LangevinSimulation,
    OverdampedSimulation,
)
from mlcg.simulation.parallel_tempering import PTSimulation
from mlcg.nn.test_outs import ASE_prior_model
from mlcg.data.atomic_data import AtomicData
from mlcg.data._keys import MASS_KEY, POSITIONS_KEY, ATOM_TYPE_KEY

torch_pi = torch.tensor(np.pi)

# def ASE_prior_model():
def model_builder(
    mol: str = "CH3CH2NH2", sum_out: bool = True, n_frame: int = 1000
):
    """Fixture that returns a simple prior-only model of
    an ASE molecule with HarmonicBonds and HarmonicAngles
    priors whose parameters are estimated from coordinates
    artifically perturbed by some small Gaussian noise.

    Parameters
    ----------
    mol:
        Molecule specifying string found in ase.build.molecule
        associated with the g2 organic molecule database
    sum_out:
        If True, the model constituents are wrapped within
        a SumOut instance

    Returns
    -------
    model_with_data:
        Dictionary that contains the fitted prior-only model
        under the key "model" and the noisey data, as a collated
        AtomicData instance, used for fitting under the key
        "collated_prior_data"
    """

    # Seeding
    rng = np.random.default_rng(934876)

    # Physical units
    temperature = 350  # K
    #:Boltzmann constan in kcal/mol/K
    kB = 0.0019872041
    beta = 1 / (temperature * kB)

    # Here we make a simple prior-only model of aluminum-fluoride
    # mol = molecule("AlF3")
    # Implement molecule with dihedrals
    mol = molecule(mol)
    test_topo = Topology.from_ase(mol)

    # Add in molecule with dihedral and compute edges
    conn_mat = get_connectivity_matrix(test_topo)
    dihedral_paths = get_n_paths(conn_mat, n=4)
    test_topo.dihedrals_from_edge_index(dihedral_paths)

    n_atoms = len(test_topo.types)
    initial_coords = np.array(mol.get_positions())

    prior_data_frames = []
    for i in range(n_frame):
        perturbed_coords = initial_coords + 0.2 * rng.standard_normal(
            initial_coords.shape
        )
        prior_data_frames.append(torch.tensor(perturbed_coords))
    prior_data_frames = torch.stack(prior_data_frames, dim=0)

    # Set up some data with bond/angle neighborlists:
    bond_edges = test_topo.bonds2torch()
    angle_edges = test_topo.angles2torch()
    dihedral_edges = test_topo.dihedrals2torch()
    non_bonded_edges = test_topo.fully_connected2torch()

    # Generete some noisy data for the priors
    nls_tags = ["bonds", "angles", "dihedrals", "repulsion"]
    nls_orders = [2, 3, 4, 2]
    nls_edges = [bond_edges, angle_edges, dihedral_edges, non_bonded_edges]

    neighbor_lists = {
        tag: make_neighbor_list(tag, order, edge_list)
        for (tag, order, edge_list) in zip(nls_tags, nls_orders, nls_edges)
    }

    prior_data_list = []
    for frame in range(prior_data_frames.shape[0]):
        data_point = AtomicData(
            pos=prior_data_frames[frame].float(),
            atom_types=torch.tensor(test_topo.types),
            masses=torch.tensor(mol.get_masses()).float(),
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
    prior_cls = [HarmonicBonds, HarmonicAngles, Dihedral, Repulsion]
    priors, stats = fit_baseline_models(collated_prior_data, beta, prior_cls)

    # Construct the model
    priors = torch.nn.ModuleDict(
        {
            name: GradientsOut(priors[name], targets=[FORCE_KEY])
            for name in priors.keys()
        }
    )

    if sum_out:
        full_model = SumOut(priors)
    else:
        full_model = priors

    model_with_data = {
        "model": full_model,
        "collated_prior_data": collated_prior_data,
        "data_list": prior_data_list,
        "molecule": mol,
        "num_examples": len(prior_data_list),
        "neighbor_lists": neighbor_lists,
    }
    return model_with_data

    # return _model_builder


# def get_initial_data():
def data_list_builder(
    mol: Atoms,
    nls: Dict,
    corruptor: Callable = None,
    add_masses=True,
) -> List[AtomicData]:
    """Helper function to generate broken data lists

    Parameters
    ----------
    mol:
        ASE molecule
    nls:
        Neighbor list dictionary
    corruptor:
        Anonynous (lambda) function that takes the current
        frame of the data list and conditionally returns
        different values. If corruptor is None, the returned
        data list will be assembled correctly.
    add_masses:
        If True, masses are specified in each AtomicData instance
        according to the ASE molecule

    Returns
    -------
    initial_data_list:
        List of AtomicData instances that has been corrupted
        at the frame and with the damage specified by the
        the corruptor. If there is no corruptor, then the data
        list will be properly constructed.
    """

    input_masses = lambda x: torch.tensor(mol.get_masses()) if x else None

    initial_data_list = []
    for frame in range(5):
        data_point = AtomicData(
            pos=torch.tensor(mol.get_positions()),
            atom_types=torch.tensor(mol.get_atomic_numbers()),
            masses=input_masses(add_masses),
            cell=None,
            velocities=None,
            neighbor_list=nls,
        )
        initial_data_list.append(data_point)

    if corruptor != None:
        # corrupt a frame
        for frame in range(5):
            corrupted_data, corrupted_key = corruptor(frame, mol)
            initial_data_list[frame][corrupted_key] = corrupted_data
    return initial_data_list

    # return data_list_builder


def test_simulation_run(
    model_dict,
    data_list_builder,
    add_masses,
):
    """Test to make sure the simulation runs"""
    data_dictionary = model_dict()
    full_model = data_dictionary["model"]
    mol = data_dictionary["molecule"]
    neighbor_lists = data_dictionary["neighbor_lists"]
    initial_data_list = data_list_builder(
        mol, neighbor_lists, add_masses=add_masses
    )

    n_timesteps = 200
    save_interval = 10
    export_interval = 10
    log_interval = 10
    dt = 0.005
    friction = 1.0
    beta = 1.0
    # with tempfile.TemporaryDirectory() as tmp:
    # filename = tmp + "/my_sim_coords_000.npy"
    filename = "test"
    open(filename, "w").close()
    simulation = LangevinSimulation(
        friction=friction,
        dt=dt,
        n_timesteps=n_timesteps,
        save_interval=save_interval,
        create_checkpoints=True,
        export_interval=export_interval,
        log_interval=log_interval,
        filename=filename,
        dtype="single",
        device="cpu",
    )
    simulation.attach_model_and_configurations(
        full_model, initial_data_list, beta
    )
    simulation.simulate()


test_simulation_run(
    model_builder,
    data_list_builder,
    True,
)
