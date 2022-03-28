# This code is adapted from https://github.com/coarse-graining/cgnet
# Authors: Brooke Husic, Nick Charron, Jiang Wang
# Contributors: Dominik Lemm, Andreas Kraemer

from typing import List, Optional, Tuple, Union, Callable
import torch
import numpy as np

from torch_geometric.data.collate import collate
import os
import time
from copy import deepcopy

from ..utils import tqdm

from ..data.atomic_data import AtomicData
from ..data._keys import ENERGY_KEY, FORCE_KEY, MASS_KEY, VELOCITY_KEY


# Physical Constants
KBOLTZMANN = 1.38064852e-23  # Boltzmann's constant in Joules/Kelvin
AVOGADRO = 6.022140857e23  # Dimensionaless Avogadro's number
JPERKCAL = 4184  # Ratio of Joules/kilocalorie


class _Simulation(object):
    """
    Parameters
    ----------
    dt : float, default=5e-4
        The integration time step for Langevin dynamics.
    beta : float, default=1
        The thermodynamic inverse temperature, :math:`1/(k_B T)`, for Boltzman
        constant :math:`k_B` and temperature :math:`T`.
    save_forces : bool, default=False
        Whether to save forces at the same saved interval as the simulation
        coordinates
    save_potential : bool, default=False
        Whether to save potential at the same saved interval as the simulation
        coordinates
    n_timesteps : int, default=100
        The length of the simulation in simulation timesteps
    save_interval : int, default=10
        The interval at which simulation timesteps should be saved. Must be
        a factor of the simulation length
    random_seed : int, default=None
        Seed for random number generator; if seeded, results always will be
        identical for the same random seed
    device : str, default='cpu'
        Device upon which simulation compuation will be carried out
    export_interval : int, default=None
        If not None, .npy files will be saved. If an int is given, then
        the int specifies at what intervals numpy files will be saved per
        observable. This number must be an integer multiple of save_interval.
        All output files should be the same shape. Forces and potentials will
        also be saved according to the save_forces and save_potential
        arguments, respectively. If friction is not None, kinetic energies
        will also be saved. This method is only implemented for a maximum of
        1000 files per observable due to file naming conventions.
    log_interval : int, default=None
        If not None, a log will be generated indicating simulation start and
        end times as well as completion updates at regular intervals. If an
        int is given, then the int specifies how many log statements will be
        output. This number must be a multiple of save_interval.
    log_type : str, default='write'
        Only relevant if log_interval is not None. If 'print', a log statement
        will be printed. If 'write', the log will be written to a .txt file.
    filename : str, default=None
        Specifies the location to which numpys and/or log files are saved.
        Must be provided if export_interval is not None and/or if log_interval
        is not None and log_type is 'write'. This provides the base file name;
        for numpy outputs, '_coords_000.npy' or similar is added. For log
        outputs, '_log.txt' is added.
    sim_subroutine :
        Optional subroutine to run at at the interval specified by
        subroutine_interval after simulation updates. The subroutine should
        take only the internal collated `AtomicData` instance as an argument.
    sim_subroutine_interval :
        Specifies the interval, in simulation steps, between successive calls to
        the subroutine, if specified.
    save_subroutine :
        Specifies additional saving procedures for extra information at the
        same interval as export_interval. The subroutine should take only the
        internal collated `AtomicData` and the current timestep // save_interval as
        arguments.
    """

    def __init__(
        self,
        dt: float = 5e-4,
        beta: Union[float, List[float]] = 1.0,
        save_forces: bool = False,
        save_energies: bool = False,
        n_timesteps: int = 100,
        save_interval: int = 10,
        random_seed: Optional[int] = None,
        device: str = "cpu",
        export_interval: Optional[int] = None,
        log_interval: Optional[int] = None,
        log_type: str = "write",
        filename: Optional[str] = None,
        specific_setup: Optional[Callable] = None,
        sim_subroutine: Optional[Callable] = None,
        sim_subroutine_interval: Optional[int] = None,
        save_subroutine: Optional[Callable] = None,
    ):
        self.model = None
        self.initial_data = None
        self.save_forces = save_forces
        self.save_energies = save_energies
        self.n_timesteps = n_timesteps
        self.save_interval = save_interval
        self.dt = dt

        self._beta_list = beta
        self.device = torch.device(device)
        self.export_interval = export_interval
        self.log_interval = log_interval

        if log_type not in ["print", "write"]:
            raise ValueError("log_type can be either 'print' or 'write'")
        self.log_type = log_type
        self.filename = filename
        self.sim_subroutine = sim_subroutine
        self.sim_subroutine_interval = sim_subroutine_interval
        self.save_subroutine = save_subroutine

        # check to make sure input options for the simulation
        self.input_option_checks()

        if random_seed is None:
            self.rng = None
        else:
            self.rng = torch.Generator(device=self.device).manual_seed(
                random_seed
            )
        self.random_seed = random_seed
        self._simulated = False

    def attach_model(self, model: torch.nn.Module):
        """setup the model to use in the simulation

        Parameters
        ----------
        model : torch.nn.Module
            Trained model used to generate simulation data
        """
        model = model.eval().to(device=self.device)
        self.model = model

    def attach_configurations(self, configurations: List[AtomicData]):
        """Setup the starting atomic configurations.

        Parameters
        ----------
        configurations : List[AtomicData]
            List of AtomicData instances representing initial structures for
        parallel simulations.
        """
        self.validate_data_list(configurations)
        self.initial_data = self.collate(configurations).to(device=self.device)
        self.n_sims = len(configurations)
        self.n_atoms = len(configurations[0].atom_types)
        self.n_dims = configurations[0].pos.shape[1]
        if isinstance(self._beta_list, float):
            self.beta = torch.tensor(self.n_sims * [self._beta_list]).to(
                self.device
            )
        else:
            self.beta = torch.tensor(self._beta_list).to(self.device)

    def simulate(self, overwrite: bool = False) -> np.ndarray:
        """Generates independent simulations.

        Parameters
        ----------
        overwrite :
            Set to True if you wish to overwrite any saved simulation data

        Returns
        -------
        simulated_coords :
            Shape [n_simulations, n_frames, n_atoms, n_dimensions]
            Also an attribute; stores the simulation coordinates at the
            save_interval
        """

        self._set_up_simulation(overwrite)
        data = deepcopy(self.initial_data)
        data.to(self.device)
        _, forces = self.calculate_potential_and_forces(data)
        for t in tqdm(range(self.n_timesteps), desc="Simulation timestep"):
            # step forward in time
            data, potential, forces = self.timestep(data, forces)

            # save to arrays if relevant
            if (t + 1) % self.save_interval == 0:

                # save arrays
                self.save(
                    data=data,
                    forces=forces,
                    potential=potential,
                    t=t,
                )

                # write numpys to file if relevant; this can be indented here because
                # it only happens when time points are also recorded
                if self.export_interval is not None:
                    if (t + 1) % self.export_interval == 0:
                        self.write((t + 1) // self.save_interval)
                        if self.save_subroutine is not None:
                            self.save_subroutine(
                                data, (t + 1) // self.save_interval
                            )

                # log if relevant; this can be indented here because
                # it only happens when time when time points are also recorded
                if self.log_interval is not None:
                    if int((t + 1) % self.log_interval) == 0:
                        self.log((t + 1) // self.save_interval)

            if self.sim_subroutine != None:
                if (t + 1) % self.sim_subroutine_interval == 0:
                    data = self.sim_subroutine(data)

            # reset data outputs to collect the new forces/energies
            data.out = {}

        # if relevant, save the remainder of the simulation
        if self.export_interval is not None:
            if int(t + 1) % self.export_interval > 0:
                self.write(t + 1)

        # if relevant, log that simulation has been completed
        if self.log_interval is not None:
            self.summary()

        self.reshape_output()

        self._simulated = True

        return self.simulated_coords

    def log(self, iter_: int):
        """Utility to print log statement or write it to an text file"""
        printstring = "{}/{} time points saved ({})".format(
            iter_, self.n_timesteps // self.save_interval, time.asctime()
        )

        if self.log_type == "print":
            print(printstring)

        elif self.log_type == "write":
            printstring += "\n"
            file = open(self._log_file, "a")
            file.write(printstring)
            file.close()

    def summary(self):
        """Prints summary information after finishing the simulation"""
        printstring = "Done simulating ({})".format(time.asctime())
        if self.log_type == "print":
            print(printstring)
        elif self.log_type == "write":
            printstring += "\n"
            with open(self._log_file, "a") as lfile:
                lfile.write(printstring)

    def calculate_potential_and_forces(
        self, data: AtomicData
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to calculate predicted forces by forwarding the current
        coordinates through self.model.

        Parameters
        ----------
        data :
            collated AtomicData instance from the previous timestep

        Returns
        -------
        potential :
            scalar potential predicted by the model
        forces :
            vector forces predicted by the model
        """

        data = self.model(data)
        potential = data.out[ENERGY_KEY].detach()
        forces = data.out[FORCE_KEY]
        return potential, forces

    def timestep(self):
        raise NotImplementedError

    @staticmethod
    def validate_data_list(data_list: List[AtomicData]):
        """Helper method to check and collate the initial data list"""

        pos_shape = data_list[0].pos.shape
        atom_types = data_list[0].atom_types
        nls = data_list[0].neighbor_list
        if MASS_KEY not in data_list[0]:
            initial_masses = False
        else:
            initial_masses = True

        # check to make sure every structure has the same number of atoms
        # and the proper neighbor_list structure
        for frame, data in enumerate(data_list):
            current_nls = data.neighbor_list
            if data.pos.shape != pos_shape:
                raise ValueError(
                    "Postions shape {} at frame {} differes from shape {} "
                    "in previous frames.".format(
                        data.pos.shape, frame, pos_shape
                    )
                )
            if (
                np.testing.assert_array_equal(
                    data.atom_types.numpy(), atom_types.numpy()
                )
                == False
            ):
                raise ValueError(
                    "Atom types {} at frame {} are not equal to atom types in "
                    "previous frames.".format(data.atom_types, frame)
                )
            if set(current_nls.keys()) != set(nls.keys()):
                raise ValueError(
                    "Neighbor list keyset {} at frame {} does not match keysets "
                    "of previous frames.".format(
                        set(data.neighbor_list.keys()), frame
                    )
                )
            for key in current_nls.keys():
                mapping = current_nls[key]["index_mapping"]
                if (
                    np.testing.assert_array_equal(
                        mapping.numpy(), nls[key]["index_mapping"]
                    )
                    == False
                ):
                    raise ValueError(
                        "Index mapping {} for key {} at frame {} does not match "
                        "those of previous frames.".format(mapping, key, frame)
                    )
            if MASS_KEY in data and initial_masses == False:
                raise ValueError(
                    "Masses {} supplied for frame {}, but previous frames "
                    "have no masses.".format(data.masses, frame)
                )
            if initial_masses == None and MASS_KEY not in data:
                raise ValueError(
                    "Masses are none for frame {}, but previous frames "
                    "have masses {}.".format(frame, data.masses)
                )
            if MASS_KEY in data:
                if data.masses.shape != atom_types.shape:
                    raise ValueError(
                        "Number of masses {} at frame {} do not match number of atoms "
                        "in previous frames.".format(
                            data.masses.shape[0], atom_types.shape[0]
                        )
                    )

    @staticmethod
    def collate(data_list: List[AtomicData]) -> AtomicData:
        """Method for collating a list of individual AtomicData instances into a
        single AtomicData instance, with proper incrementing of AtomicData properties.
        """

        collated_data, _, _ = collate(
            data_list[0].__class__,
            data_list=data_list,
            increment=True,
            add_batch=True,
        )
        return collated_data

    def input_option_checks(self):
        """Method to catch any problems before starting a simulation:
        - Make sure the save_interval evenly divides the simulation length
        - Checks compatibility of arguments to save and log
        - Sets up saving parameters for numpy and log files, if relevant
        """

        # make sure save interval is a factor of total n_timesteps
        if self.n_timesteps % self.save_interval != 0:
            raise ValueError(
                "The save_interval must be a factor of the simulation n_timesteps"
            )

        # check whether a directory is specified if any saving is done
        if self.export_interval is not None and self.filename is None:
            raise RuntimeError(
                "Must specify filename if export_interval isn't None"
            )
        if self.log_interval is not None:
            if self.log_type == "write" and self.filename is None:
                raise RuntimeError(
                    "Must specify filename if log_interval isn't None and log_type=='write'"
                )

        # saving numpys
        if self.export_interval is not None:
            if self.n_timesteps // self.export_interval >= 1000:
                raise ValueError(
                    "Simulation saving is not implemented if more than 1000 files will be generated"
                )

            if os.path.isfile("{}_coords_000.npy".format(self.filename)):
                raise ValueError(
                    "{} already exists; choose a different filename.".format(
                        "{}_coords_000.npy".format(self.filename)
                    )
                )

            if self.export_interval is not None:
                if self.export_interval % self.save_interval != 0:
                    raise ValueError(
                        "Numpy saving must occur at a multiple of save_interval"
                    )
                self._npy_file_index = 0
                self._npy_starting_index = 0

        # logging
        if self.log_interval is not None:
            if self.log_interval % self.save_interval != 0:
                raise ValueError(
                    "Logging must occur at a multiple of save_interval"
                )

            if self.log_type == "write":
                self._log_file = self.filename + "_log.txt"

                if os.path.isfile(self._log_file):
                    raise ValueError(
                        "{} already exists; choose a different filename.".format(
                            self._log_file
                        )
                    )
        # simulation subroutine
        if self.sim_subroutine != None and self.sim_subroutine_interval == None:
            raise ValueError(
                "subroutine {} specified, but subroutine_interval is ambiguous.".format(
                    self.sim_subroutine
                )
            )
        if self.sim_subroutine_interval != None and self.sim_subroutine == None:
            raise ValueError(
                "subroutine interval specified, but subroutine is ambiguous."
            )

    def _get_numpy_count(self):
        """Returns a string 000-999 for appending to numpy file outputs"""
        return f"{self._npy_file_index:03d}"

    def _swap_and_export(
        self, input_tensor: torch.Tensor, axis1: int = 0, axis2: int = 1
    ) -> np.ndarray:
        """Helper method to exchange the zeroth and first axes of tensors that
        will be output or exported as numpy arrays

        Parameters
        ----------
        input_tensor:
            Tensor of shape (n_save_steps, n_sims, n_atoms, n_dims)
        axis1:
            The axis that will be occupied by data from axis2 after the swap
        axis2:
            The axis that will be occupied by data from axis1 after the swap

        Returns
        -------
        swapped_data:
            Numpy array of the input data with swapped axes
        """

        axes = list(range(len(input_tensor.size())))
        axes[axis1] = axis2
        axes[axis2] = axis1
        swapped_data = input_tensor.permute(*axes)
        return swapped_data.cpu().detach().numpy()

    def _set_up_simulation(self, overwrite: bool = False):
        """Method to setup up saving and logging options"""
        if self._simulated and not overwrite:
            raise RuntimeError(
                "Simulation results are already populated. "
                "To rerun, set overwrite=True."
            )

        self._save_size = int(self.n_timesteps / self.save_interval)

        self.simulated_coords = torch.zeros(
            (self._save_size, self.n_sims, self.n_atoms, self.n_dims)
        )
        if self.save_forces:
            self.simulated_forces = torch.zeros(
                (self._save_size, self.n_sims, self.n_atoms, self.n_dims)
            )
        else:
            self.simulated_forces = None

        if self.save_energies:
            self.simulated_potential = torch.zeros(self._save_size, self.n_sims)
        else:
            self.simulated_potential = None

        if self.log_interval is not None:
            printstring = "Generating {} simulations of n_timesteps {} saved at {}-step intervals ({})".format(
                self.n_sims,
                self.n_timesteps,
                self.save_interval,
                time.asctime(),
            )
            if self.log_type == "print":
                print(printstring)

            elif self.log_type == "write":
                printstring += "\n"
                file = open(self._log_file, "a")
                file.write(printstring)
                file.close()

    def save(
        self,
        data: AtomicData,
        forces: torch.Tensor,
        potential: torch.Tensor,
        t: int,
    ):
        """Utility to store saved values of coordinates and, if relevant,
        also forces, potential, and/or kinetic energy
        Parameters
        ----------
        x_new :
            current coordinates
        forces:
            current forces
        potential :
            current potential
        t :
            current timestep
        """
        x_new = data.pos.view(-1, self.n_atoms, self.n_dims)
        forces = forces.view(-1, self.n_atoms, self.n_dims)

        save_ind = t // self.save_interval

        self.simulated_coords[save_ind, :, :] = x_new

        if self.save_forces:
            self.simulated_forces[save_ind, :, :] = forces

        if self.save_energies:
            if self.simulated_potential is None:
                assert potential.shape[0] == self.n_sims
                potential_dims = [self._save_size, self.n_sims] + [
                    potential.shape[j] for j in range(1, len(potential.shape))
                ]
                self.simulated_potential = torch.zeros((potential_dims))

            self.simulated_potential[t // self.save_interval] = potential

    def write(self, iter_: int):
        """Utility to write numpy arrays to disk"""
        key = self._get_numpy_count()

        coords_to_export = self.simulated_coords[
            self._npy_starting_index : iter_
        ]
        coords_to_export = self._swap_and_export(coords_to_export)
        np.save("{}_coords_{}.npy".format(self.filename, key), coords_to_export)

        if self.save_forces:
            forces_to_export = self.simulated_forces[
                self._npy_starting_index : iter_
            ]
            forces_to_export = self._swap_and_export(forces_to_export)
            np.save(
                "{}_forces_{}.npy".format(self.filename, key), forces_to_export
            )

        if self.save_energies:
            potentials_to_export = self.simulated_potential[
                self._npy_starting_index : iter_
            ]
            potentials_to_export = self._swap_and_export(potentials_to_export)
            np.save(
                "{}_potential_{}.npy".format(self.filename, key),
                potentials_to_export,
            )

        self._npy_starting_index = iter_
        self._npy_file_index += 1

    def reshape_output(self):
        # reshape output attributes
        self.simulated_coords = self._swap_and_export(self.simulated_coords)

        if self.save_forces:
            self.simulated_forces = self._swap_and_export(self.simulated_forces)

        if self.save_energies:
            self.simulated_potential = self._swap_and_export(
                self.simulated_potential
            )
