from typing import List, Callable, Dict
import tempfile
import torch
import pytest
from ase.atoms import Atoms

from mlcg.simulation.base import _Simulation
from mlcg.simulation.langevin import LangevinSimulation, OverdampedSimulation
from mlcg.nn.test_outs import ASE_prior_model
from mlcg.data.atomic_data import AtomicData
from mlcg.data._keys import MASS_KEY, POSITIONS_KEY, ATOM_TYPE_KEY


@pytest.fixture
def get_initial_data():
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

    return data_list_builder


### corruptors - lambdas that introduce a problem in the data list ###

# Puts the wrong mass on the fourth frame
wrong_mass_fn = (
    lambda frame, mol: (2 * torch.tensor(mol.get_masses()), MASS_KEY)
    if frame == 3
    else (torch.tensor(mol.get_masses()), MASS_KEY)
)

# Gives a structure with the wrong shape on the third frame
wrong_pos_fn = (
    lambda frame, mol: (torch.randn(7, 3), POSITIONS_KEY)
    if frame == 2
    else (torch.tensor(mol.get_positions()), POSITIONS_KEY)
)
# Gives the wrong atomic types on the second frame
wrong_atom_type_fn = (
    lambda frame, mol: (
        7 * torch.tensor(mol.get_atomic_numbers()),
        ATOM_TYPE_KEY,
    )
    if frame == 1
    else (torch.tensor(mol.get_atomic_numbers()), ATOM_TYPE_KEY)
)


@pytest.mark.parametrize(
    "ASE_prior_model, get_initial_data, corruptor, add_masses, expected_raise",
    [
        (
            # Should raise error: one frame has different masses
            ASE_prior_model,
            get_initial_data,
            wrong_mass_fn,
            True,
            ValueError,
        ),
        (
            # Should raise error: one frame has a different structure
            ASE_prior_model,
            get_initial_data,
            wrong_pos_fn,
            True,
            ValueError,
        ),
        (
            # Should raise error: one frame has a different atom types
            ASE_prior_model,
            get_initial_data,
            wrong_atom_type_fn,
            True,
            ValueError,
        ),
    ],
    indirect=["ASE_prior_model", "get_initial_data"],
)
def test_data_list_raises(
    ASE_prior_model, get_initial_data, corruptor, add_masses, expected_raise
):
    """Test to make sure certain warnings/errors are raised regarding the data list"""
    data_dictionary = ASE_prior_model()
    full_model = data_dictionary["model"]
    mol = data_dictionary["molecule"]
    neighbor_lists = data_dictionary["neighbor_lists"]

    initial_data_list = get_initial_data(
        mol, neighbor_lists, corruptor, add_masses=add_masses
    )

    if isinstance(expected_raise, Exception):
        with pytest.raises(expected_raise):
            simulation = _Simulation()
            simulation.attach_configurations(initial_data_list)
    if isinstance(expected_raise, UserWarning):
        with pytest.warns(expected_raise):
            simulation = _Simulation()
            simulation.attach_configurations(initial_data_list)


@pytest.mark.parametrize(
    "ASE_prior_model, get_initial_data, add_masses, sim_class, sim_args, sim_kwargs",
    [
        (
            ASE_prior_model,
            get_initial_data,
            False,
            OverdampedSimulation,
            [],
            {},
        ),
        (
            ASE_prior_model,
            get_initial_data,
            True,
            LangevinSimulation,
            [1.0],
            {},
        ),
    ],
    indirect=["ASE_prior_model", "get_initial_data"],
)
def test_simulation_run(
    ASE_prior_model,
    get_initial_data,
    add_masses,
    sim_class,
    sim_args,
    sim_kwargs,
):
    """Test to make sure the simulation runs"""
    data_dictionary = ASE_prior_model()
    full_model = data_dictionary["model"]
    mol = data_dictionary["molecule"]
    neighbor_lists = data_dictionary["neighbor_lists"]
    initial_data_list = get_initial_data(
        mol, neighbor_lists, corruptor=None, add_masses=add_masses
    )

    simulation = sim_class(*sim_args, **sim_kwargs)
    simulation.attach_configurations(initial_data_list)
    simulation.attach_model(full_model)
    simulation.simulate()


@pytest.mark.parametrize(
    "ASE_prior_model, get_initial_data, add_masses, sim_class, sim_args, sim_kwargs",
    [
        (
            ASE_prior_model,
            get_initial_data,
            False,
            OverdampedSimulation,
            [],
            {},
        ),
        (
            ASE_prior_model,
            get_initial_data,
            True,
            LangevinSimulation,
            [1.0],
            {},
        ),
    ],
    indirect=["ASE_prior_model", "get_initial_data"],
)
def test_overwrite_protection(
    ASE_prior_model,
    get_initial_data,
    add_masses,
    sim_class,
    sim_args,
    sim_kwargs,
):
    """Test to make sure that overwrite protection works"""
    data_dictionary = ASE_prior_model()
    full_model = data_dictionary["model"]
    mol = data_dictionary["molecule"]
    neighbor_lists = data_dictionary["neighbor_lists"]
    initial_data_list = get_initial_data(
        mol, neighbor_lists, corruptor=None, add_masses=add_masses
    )

    with tempfile.TemporaryDirectory() as tmp:
        filename = tmp + "/my_sim_coords_000.npy"
        open(filename, "w").close()
        sim_kwargs["filename"] = filename
        simulation = sim_class(*sim_args, **sim_kwargs)
        simulation.attach_configurations(initial_data_list)
        simulation.attach_model(full_model)
        simulation.simulate()

        with pytest.raises(RuntimeError):
            simulation.simulate()
