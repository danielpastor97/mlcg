# This code is adapted from https://github.com/coarse-graining/cgnet
# Authors: Brooke Husic, Nick Charron, Jiang Wang
# Contributors: Dominik Lemm, Andreas Kraemer

from typing import List, Tuple
import torch
import numpy as np

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data.collate import collate
import os
import time
import warnings

from tqdm import tqdm

from ..data.atomic_data import AtomicData
from ..data._keys import ENERGY_KEY, FORCE_KEY, MASS_KEY, VELOCITY_KEY


class _Simulation(object):
    """Abstract simulation class"""

    def __init__(
        self,
        model: torch.nn.Module,
        initial_data_list: List[AtomicData],
        dt: float = 5e-4,
        beta: float = 1.0,
        save_forces: bool = False,
        save_energies: bool = False,
        length: int = 100,
        save_interval: int = 10,
        random_seed: int = None,
        device: torch.device = torch.device("cpu"),
        export_interval: int = None,
        log_interval: int = None,
        log_type: str = "write",
        filename: str = None,
    ):

        self.initial_data = self.check_and_collate_data(initial_data_list)
        model.eval()
        self.model = model
        self.n_sims = len(initial_data_list)
        self.n_atoms = len(initial_data_list[0].atom_types)
        self.n_dims = initial_data_list[0].pos.shape[1]
        self.save_forces = save_forces
        self.save_energies = save_energies
        self.length = length
        self.save_interval = save_interval
        self.dt = dt
        self.beta = beta
        self.device = device
        self.export_interval = export_interval
        self.log_interval = log_interval

        if log_type not in ["print", "write"]:
            raise ValueError("log_type can be either 'print' or 'write'")
        self.log_type = log_type
        self.filename = filename
        # check to make sure input options for the simulation
        self.input_option_checks()

        if random_seed is None:
            self.rng = torch.default_generator
        else:
            self.rng = torch.Generator().manual_seed(random_seed)
        self.random_seed = random_seed
        self._simulated = False

    def check_and_collate_data(self, data_list: List[AtomicData]) -> AtomicData:
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
                    "Postions shape {} at frame {} differes from shape {} in previous frames.".format(
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
                    "Atom types {} at frame {} are not equal to atom types in previous frames.".format(
                        data.atom_types, frame
                    )
                )
            if set(current_nls.keys()) != set(nls.keys()):
                raise ValueError(
                    "Neighbor list keyset {} at frame {} does not match keysets of previous frames.".format(
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
                        "Index mapping {} for key {} at frame {} does not match those of previous frames.".format(
                            mapping, key, frame
                        )
                    )
            if MASS_KEY in data and initial_masses == False:
                raise ValueError(
                    "Masses {} supplied for frame {}, but previous frames have no masses.".format(
                        data.masses, frame
                    )
                )
            if initial_masses == None and MASS_KEY not in data:
                raise ValueError(
                    "Masses are none for frame {}, but previous frames have masses {}.".format(
                        frame, masses
                    )
                )
            if MASS_KEY in data:
                if data.masses.shape != atom_types.shape:
                    raise ValueError(
                        "Number of masses {} at frame {} do not match number of atoms in previous frames.".format(
                            data.masses.shape[0], atom_types.shape[0]
                        )
                    )

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

        # make sure save interval is a factor of total length
        if self.length % self.save_interval != 0:
            raise ValueError(
                "The save_interval must be a factor of the simulation length"
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
            if self.length // self.export_interval >= 1000:
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

    def log(self, iter_: int):
        """Utility to print log statement or write it to an text file"""
        printstring = "{}/{} time points saved ({})".format(
            iter_, self.length // self.save_interval, time.asctime()
        )

        if self.log_type == "print":
            print(printstring)

        elif self.log_type == "write":
            printstring += "\n"
            file = open(self._log_file, "a")
            file.write(printstring)
            file.close()

    def _get_numpy_count(self):
        """Returns a string 000-999 for appending to numpy file outputs"""
        if self._npy_file_index < 10:
            return "00{}".format(self._npy_file_index)
        elif self._npy_file_index < 100:
            return "0{}".format(self._npy_file_index)
        else:
            return "{}".format(self._npy_file_index)

    def _swap_and_export(self, data, axis1=0, axis2=1):
        """Helper method to exchange the zeroth and first axes of tensors that
        will be output or exported as numpy arrays
        """
        axes = list(range(len(data.size())))
        axes[axis1] = axis2
        axes[axis2] = axis1
        swapped_data = data.permute(*axes)
        return swapped_data.cpu().detach().numpy()

    def save(self):
        raise NotImplementedError

    def write(self):
        raise NotImplementedError

    def calculate_potential_and_forces(self):
        raise NotImplementedError

    def simulate(self):
        raise NotImplementedError

    def timestep(self):
        raise NotImplementedError


class LangevinSimulation(_Simulation):
    """Langevin simulatin class for trainen models.

    The following BAOA(F)B integration scheme is used, where:
    ```
        B = deterministic velocity update
        A = deterministic position update
        O = stochastic velocity update
        F = force calculation (i.e., from the cgnet)
    ```
    We have chosen the following implementation so as to only calculate
    forces once per timestep:

    .. math::

        F = - \nabla ( U(X_t) ) \\
        [BB] V_(t+1) = V_t + dt * \frac{F}{m} \\
        [A] X_(t+\frac{1}{2}) = X_t + V * \frac{dt}{2} \\
        [O] V_(t+1) = V_(t+1) * \text{vscale} + dW_t * \text{noisescale}
        [A] X_(t+1) = X_(t+\frac{1}{2}) + V * \frac{dt}{2}

    Where:

    .. math::

        \text{vscale} = exp(-\text{friction} * dt)
        \text{noisecale} = \sqrt(1 - \text{vscale}^2)

    A diffusion constant :math:`D` can be back-calculated using
    the Einstein relation:

    .. math::

        D = 1 / (\beta * \text{friction})

    Initial velocities are set to zero with Gaussian noise.

    Parameters
    ----------
    model :
        Trained model used to generate simulation data
    initial_data :
        List of AtomicData instances representing initial structures for
        parallel simulations.
    dt :
        The integration time step for Langevin dynamics.
    beta :
        The thermodynamic inverse temperature, :math:`1/(k_B T)`, for Boltzman
        constant :math:`k_B` and temperature :math:`T`.
    friction :
        Friction value for Langevin scheme, in units of inverse time.
    save_forces :
        Whether to save forces at the same saved interval as the simulation
        coordinates
    save_potential :
        Whether to save potential at the same saved interval as the simulation
        coordinates
    length :
        The length of the simulation in simulation timesteps
    save_interval :
        The interval at which simulation timesteps should be saved. Must be
        a factor of the simulation length
    random_seed :
        Seed for random number generator; if seeded, results always will be
        identical for the same random seed
    device :
        Device upon which simulation compuation will be carried out
    export_interval :
        If not None, .npy files will be saved. If an int is given, then
        the int specifies at what intervals numpy files will be saved per
        observable. This number must be an integer multiple of save_interval.
        All output files should be the same shape. Forces and potentials will
        also be saved according to the save_forces and save_potential
        arguments, respectively. If friction is not None, kinetic energies
        will also be saved. This method is only implemented for a maximum of
        1000 files per observable due to file naming conventions.
    log_interval :
        If not None, a log will be generated indicating simulation start and
        end times as well as completion updates at regular intervals. If an
        int is given, then the int specifies how many log statements will be
        output. This number must be a multiple of save_interval.
    log_type :
        Only relevant if log_interval is not None. If 'print', a log statement
        will be printed. If 'write', the log will be written to a .txt file.
    filename :
        Specifies the location to which numpys and/or log files are saved.
        Must be provided if export_interval is not None and/or if log_interval
        is not None and log_type is 'write'. This provides the base file name;
        for numpy outputs, '_coords_000.npy' or similar is added. For log
        outputs, '_log.txt' is added.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        initial_data_list: List[AtomicData],
        friction: float,
        dt: float = 5e-4,
        beta: float = 1.0,
        save_forces: bool = False,
        save_energies: bool = False,
        length: int = 100,
        save_interval: int = 10,
        random_seed: int = None,
        device: torch.device = torch.device("cpu"),
        export_interval: int = None,
        log_interval: int = None,
        log_type: str = "write",
        filename: str = None,
    ):

        super(LangevinSimulation, self).__init__(
            model,
            initial_data_list,
            dt=dt,
            beta=beta,
            save_forces=save_forces,
            save_energies=save_energies,
            length=length,
            save_interval=save_interval,
            random_seed=random_seed,
            device=device,
            export_interval=export_interval,
            log_interval=log_interval,
            log_type=log_type,
            filename=filename,
        )

        assert friction is not None
        assert friction > 0
        self.friction = friction

        self.vscale = np.exp(-self.dt * self.friction)
        self.noisescale = np.sqrt(1 - self.vscale * self.vscale)

    def _set_up_simulation(self, overwrite: bool = False):
        """Method to setup up saving and logging options"""
        if self._simulated and not overwrite:
            raise RuntimeError(
                "Simulation results are already populated. "
                "To rerun, set overwrite=True."
            )

        self._save_size = int(self.length / self.save_interval)

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
            self.simulated_kinetic_energies = torch.zeros(
                self._save_size, self.n_sims
            )
        else:
            self.simulated_potential = None
            self.simulated_kinetic_energies = None

        if self.log_interval is not None:
            printstring = "Generating {} simulations of length {} saved at {}-step intervals ({})".format(
                self.n_sims, self.length, self.save_interval, time.asctime()
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
        x_new: torch.Tensor,
        v_new: torch.Tensor,
        forces: torch.Tensor,
        potential: torch.tensor,
        masses: torch.Tensor,
        t: int,
    ):
        """Utilities to store saved values of coordinates and, if relevant,
        also forces, potential, and/or kinetic energy
        Parameters
        ----------
        x_new :
            current coordinates
        v_new :
            current velocities
        forces:
            current forces
        potential :
            current potential
        masses :
            atom masses for kinetic energy calculation
        t :
            current timestep
        """

        x_new = x_new.view(-1, self.n_atoms, self.n_dims)
        v_new = v_new.view(-1, self.n_atoms, self.n_dims)
        forces = forces.view(-1, self.n_atoms, self.n_dims)
        masses = masses.view(self.n_sims, self.n_atoms)

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

            kes = 0.5 * torch.sum(
                torch.sum(masses[:, :, None] * v_new ** 2, dim=2), dim=1
            )
            self.kinetic_energies[save_ind, :] = kes

    def write(self, iter_: int):
        """Utility to save numpy arrays"""
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

            kinetic_energies_to_export = self.kinetic_energies[
                self._npy_starting_index : iter_
            ]
            kinetic_energies_to_export = self._swap_and_export(
                kinetic_energies_to_export
            )
            np.save(
                "{}_kineticenergy_{}.npy".format(self.filename, key),
                kinetic_energies_to_export,
            )

        self._npy_starting_index = iter_
        self._npy_file_index += 1

    def timestep(
        self,
        x_old: torch.Tensor,
        v_old: torch.Tensor,
        forces: torch.Tensor,
        masses: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Timestep method for Langevin dynamics
        Parameters
        ----------
        x_old :
            coordinates before propagataion
        v_old :
            velocities before propagation
        forces:
            forces at x_old, before propagation

        Returns
        -------
        x_new :
            coordinates after propagation
        v_new :
            velocites after propagation
        """
        # BB (velocity update); uses whole timestxep
        v_new = v_old + self.dt * forces / masses[:, None]

        # A (position update)
        x_new = x_old + v_new * self.dt / 2.0

        # O (noise)
        noise = torch.sqrt(1.0 / self.beta / masses[:, None])
        noise = noise * torch.randn(size=x_new.size(), generator=self.rng).to(
            self.device
        )
        v_new = v_new * self.vscale
        v_new = v_new + self.noisescale * noise

        # A
        x_new = x_new + v_new * self.dt / 2.0

        return x_new, v_new

    def calculate_potential_and_forces(
        self, data_old: AtomicData
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to calculate predicted forces by forwarding the current
        coordinates through self.model.

        Parameters
        ----------
        data_old :
            collated AtomicData instance from the previous timestep

        Returns
        -------
        potential :
            scalar potential predicted by the model
        forces :
            vector forces predicted by the model
        """

        data_old = self.model(data_old)
        potential = data_old.out[ENERGY_KEY].detach()
        forces = data_old.out[FORCE_KEY]
        return potential, forces

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
        data_old = self.initial_data

        # for each simulation step
        # initialize velocities at zero
        v_old = torch.tensor(
            np.zeros(self.initial_data.pos.shape), dtype=torch.float32
        )
        data_old.velocities = v_old

        for t in tqdm(range(self.length), desc="Simulation timestep"):
            # produce potential and forces from model
            potential, forces = self.calculate_potential_and_forces(data_old)
            x_old = data_old.pos
            v_old = data_old.velocities

            # step forward in time
            x_new, v_new = self.timestep(x_old, v_old, forces, data_old.masses)

            # save to arrays if relevant
            if (t + 1) % self.save_interval == 0:

                # save arrays
                self.save(
                    x_new,
                    v_new,
                    forces,
                    potential,
                    self.initial_data[MASS_KEY],
                    t,
                )

                # write numpys to file if relevant; this can be indented here because
                # it only happens when time points are also recorded
                if self.export_interval is not None:
                    if (t + 1) % self.export_interval == 0:
                        self.write((t + 1) // self.save_interval)

                # log if relevant; this can be indented here because
                # it only happens when time when time points are also recorded
                if self.log_interval is not None:
                    if int((t + 1) % self.log_interval) == 0:
                        self.log((t + 1) // self.save_interval)

            # prepare for next timestep
            data_old.pos = x_new
            data_old.velocities = v_new

            # reset data outputs to collect the new forces/energies
            data_old.out = {}

        # if relevant, save the remainder of the simulation
        if self.export_interval is not None:
            if int(t + 1) % self.export_interval > 0:
                self.write(t + 1)

        # if relevant, log that simulation has been completed
        if self.log_interval is not None:
            printstring = "Done simulating ({})".format(time.asctime())
            if self.log_type == "print":
                print(printstring)
            elif self.log_type == "write":
                printstring += "\n"
                file = open(self._log_file, "a")
                file.write(printstring)
                file.close()

        # reshape output attributes
        self.simulated_coords = self._swap_and_export(self.simulated_coords)

        if self.save_forces:
            self.simulated_forces = self._swap_and_export(self.simulated_forces)

        if self.save_energies:
            self.simulated_potential = self._swap_and_export(
                self.simulated_potential
            )
            self.simulated_kinetic_energies = self._swap_and_export(
                self.simulated_kinetic_energies
            )

        self._simulated = True

        return self.simulated_coords


class OverdampedSimulation(_Simulation):
    """Overdamped Langevin simulation class for trained models.

    The following Brownian motion scheme is used:

    .. math::

        dX_t = - \nabla( U( X_t ) ) * D * dt + \sqrt( 2 * D * dt / \beta ) * dW_t

    for coordinates :math:`X_t` at time :math:`t`, potential energy :math:`U`,
    diffusion :math:`D`, thermodynamic inverse temperature :math:`\beta`,
    time step :math:`dt`, and stochastic Weiner process :math:`W`.

    Parameters
    ----------
    model :
        Trained model used to generate simulation data
    initial_data_list :
        List of AtomicData instances representing initial structures for
        parallel simulations.
    dt :
        The integration time step for overdamped Langevin dynamics.
    beta :
        The thermodynamic inverse temperature, :math:`1/(k_B T)`, for Boltzman
        constant :math:`k_B` and temperature :math:`T`.
    diffusion :
        The constant diffusion parameter :math:`D`. By default, the diffusion
        is set to unity and is absorbed into the :math:`dt` argument.
        However, users may specify separate diffusion and :math:`dt`
        parameters in the case that they have some estimate of the diffusion.
    save_forces :
        Whether to save forces at the same saved interval as the simulation
        coordinates
    save_potential :
        Whether to save potential at the same saved interval as the simulation
        coordinates
    length :
        The length of the simulation in simulation timesteps
    save_interval :
        The interval at which simulation timesteps should be saved. Must be
        a factor of the simulation length
    random_seed :
        Seed for random number generator; if seeded, results always will be
        identical for the same random seed
    device :
        Device upon which simulation compuation will be carried out
    export_interval :
        If not None, .npy files will be saved. If an int is given, then
        the int specifies at what intervals numpy files will be saved per
        observable. This number must be an integer multiple of save_interval.
        All output files should be the same shape. Forces and potentials will
        also be saved according to the save_forces and save_potential
        arguments, respectively. If friction is not None, kinetic energies
        will also be saved. This method is only implemented for a maximum of
        1000 files per observable due to file naming conventions.
    log_interval :
        If not None, a log will be generated indicating simulation start and
        end times as well as completion updates at regular intervals. If an
        int is given, then the int specifies how many log statements will be
        output. This number must be a multiple of save_interval.
    log_type :
        Only relevant if log_interval is not None. If 'print', a log statement
        will be printed. If 'write', the log will be written to a .txt file.
    filename :
        Specifies the location to which numpys and/or log files are saved.
        Must be provided if export_interval is not None and/or if log_interval
        is not None and log_type is 'write'. This provides the base file name;
        for numpy outputs, '_coords_000.npy' or similar is added. For log
        outputs, '_log.txt' is added.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        initial_data_list: List[AtomicData],
        diffusion: float = 1.0,
        dt: float = 5e-4,
        beta: float = 1.0,
        save_forces: bool = False,
        save_energies: bool = False,
        length: int = 100,
        save_interval: int = 10,
        random_seed: int = None,
        device: torch.device = torch.device("cpu"),
        export_interval: int = None,
        log_interval: int = None,
        log_type: str = "write",
        filename: str = None,
    ):

        super(OverdampedSimulation, self).__init__(
            model,
            initial_data_list,
            dt=dt,
            beta=beta,
            save_forces=save_forces,
            save_energies=save_energies,
            length=length,
            save_interval=save_interval,
            random_seed=random_seed,
            device=device,
            export_interval=export_interval,
            log_interval=log_interval,
            log_type=log_type,
            filename=filename,
        )

        assert diffusion is not None
        assert diffusion > 0
        self.diffusion = diffusion
        self._dtau = self.diffusion * self.dt

        if MASS_KEY in self.initial_data:
            warnings.warn(
                "Masses were provided, but will not be used since "
                "an overdamped Langevin scheme is being used for integration."
            )

    def _set_up_simulation(self, overwrite: bool = False):
        """Method to setup up saving and logging options"""
        if self._simulated and not overwrite:
            raise RuntimeError(
                "Simulation results are already populated. "
                "To rerun, set overwrite=True."
            )

        self._save_size = int(self.length / self.save_interval)

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
            printstring = "Generating {} simulations of length {} saved at {}-step intervals ({})".format(
                self.n_sims, self.length, self.save_interval, time.asctime()
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
        x_new: torch.Tensor,
        forces: torch.Tensor,
        potential: torch.tensor,
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
        x_new = x_new.view(-1, self.n_atoms, self.n_dims)
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

    def timestep(
        self,
        x_old: torch.Tensor,
        forces: torch.Tensor,
    ) -> torch.Tensor:
        """Timestep method for Langevin dynamics
        Parameters
        ----------
        x_old :
            coordinates before propagataion
        forces:
            forces at x_old, before propagation

        Returns
        -------
        x_new :
            coordinates after propagation
        """

        noise = torch.randn(size=x_old.size(), generator=self.rng).to(
            self.device
        )
        x_new = (
            x_old.detach()
            + forces * self._dtau
            + np.sqrt(2 * self._dtau / self.beta) * noise
        )
        return x_new

    def calculate_potential_and_forces(
        self, data_old: AtomicData
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to calculate predicted forces by forwarding the current
        coordinates through self.model.

        Parameters
        ----------
        data_old :
            collated AtomicData instance from the previous timestep

        Returns
        -------
        potential :
            scalar potential predicted by the model
        forces :
            vector forces predicted by the model
        """

        data_old = self.model(data_old)
        potential = data_old.out[ENERGY_KEY].detach()
        forces = data_old.out[FORCE_KEY]
        return potential, forces

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
        data_old = self.initial_data

        for t in tqdm(range(self.length), desc="Simulation timestep"):
            # produce potential and forces from model
            potential, forces = self.calculate_potential_and_forces(data_old)
            x_old = data_old.pos

            # step forward in time
            x_new = self.timestep(x_old, forces)

            # save to arrays if relevant
            if (t + 1) % self.save_interval == 0:

                # save arrays
                self.save(
                    x_new,
                    forces,
                    potential,
                    t,
                )

                # write numpys to file if relevant; this can be indented here because
                # it only happens when time points are also recorded
                if self.export_interval is not None:
                    if (t + 1) % self.export_interval == 0:
                        self._save_numpy((t + 1) // self.save_interval)

                # log if relevant; this can be indented here because
                # it only happens when time when time points are also recorded
                if self.log_interval is not None:
                    if int((t + 1) % self.log_interval) == 0:
                        self.log((t + 1) // self.save_interval)

            # prepare for next timestep
            data_old.pos = x_new

            # reset data outputs to collect the new forces/energies
            data_old.out = {}

        # if relevant, save the remainder of the simulation
        if self.export_interval is not None:
            if int(t + 1) % self.export_interval > 0:
                self.write(t + 1)

        # if relevant, log that simulation has been completed
        if self.log_interval is not None:
            printstring = "Done simulating ({})".format(time.asctime())
            if self.log_type == "print":
                print(printstring)
            elif self.log_type == "write":
                printstring += "\n"
                file = open(self._log_file, "a")
                file.write(printstring)
                file.close()

        # reshape output attributes
        self.simulated_coords = self._swap_and_export(self.simulated_coords)

        if self.save_forces:
            self.simulated_forces = self._swap_and_export(self.simulated_forces)

        if self.save_energies:
            self.simulated_potential = self._swap_and_export(
                self.simulated_potential
            )

        self._simulated = True

        return self.simulated_coords
