from typing import List
import torch
import numpy as np
import warnings

from ..data.atomic_data import AtomicData
from ..data._keys import MASS_KEY, VELOCITY_KEY
from .base import _Simulation


class LangevinSimulation(_Simulation):
    r"""Langevin simulatin class for trained models.

    The following (F)BBAOA integration scheme is used, where:

    .. code-block::python

        B = deterministic velocity update
        A = deterministic position update
        O = stochastic velocity update
        F = force calculation (i.e., from the cgnet)


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

        assert friction > 0
        self.friction = friction

        self.vscale = np.exp(-self.dt * self.friction)
        self.noisescale = np.sqrt(1 - self.vscale * self.vscale)

        if VELOCITY_KEY not in self.initial_data:
            # initialize velocities at zero
            self.initial_data.velocities = torch.zeros_like(
                self.initial_data.pos
            )

        assert self.initial_data.velocities.shape == self.initial_data.pos.shape

    def timestep(
        self,
        data: AtomicData,
        forces: torch.Tensor,
    ) -> AtomicData:
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
        v_old = data.velocities
        masses = data.masses
        x_old = data.pos
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

        data.pos = x_new
        data.velocities = v_new
        return data

    def _set_up_simulation(self, overwrite: bool = False):
        """Method to setup up saving and logging options"""
        super()._set_up_simulation(overwrite)

        if self.save_energies:
            self.simulated_kinetic_energies = torch.zeros(
                self._save_size, self.n_sims
            )
        else:
            self.simulated_kinetic_energies = None

    def save(
        self,
        data: AtomicData,
        forces: torch.Tensor,
        potential: torch.tensor,
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
        super().save(data, forces, potential, t)

        v_new = data.velocities.view(-1, self.n_atoms, self.n_dims)
        masses = data.masses.view(self.n_sims, self.n_atoms)

        save_ind = t // self.save_interval

        if self.save_energies:
            kes = 0.5 * torch.sum(
                torch.sum(masses[:, :, None] * v_new ** 2, dim=2), dim=1
            )
            self.kinetic_energies[save_ind, :] = kes

    def write(self, iter_: int):
        """Utility to save numpy arrays"""
        key = self._get_numpy_count()
        if self.save_energies:
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

        super().write(iter_)

    def reshape_output(self):
        super().reshape_output()
        if self.save_energies:
            self.simulated_kinetic_energies = self._swap_and_export(
                self.simulated_kinetic_energies
            )


class OverdampedSimulation(_Simulation):
    r"""Overdamped Langevin simulation class for trained models.

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

    def timestep(self, data: AtomicData, forces: torch.Tensor) -> AtomicData:
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
        x_old = data.pos
        noise = torch.randn(size=x_old.size(), generator=self.rng).to(
            self.device
        )
        x_new = (
            x_old.detach()
            + forces * self._dtau
            + np.sqrt(2 * self._dtau / self.beta) * noise
        )
        data.pos = x_new
        return data
