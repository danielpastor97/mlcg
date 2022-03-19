from typing import List, Tuple, Any, Dict
import torch
import numpy as np
import warnings
from copy import deepcopy

from ..data.atomic_data import AtomicData
from ..data._keys import (
    MASS_KEY,
    VELOCITY_KEY,
    BETA_KEY,
    POSITIONS_KEY,
    ENERGY_KEY,
)
from .base import _Simulation


class LangevinSimulation(_Simulation):
    r"""Langevin simulatin class for trained models.

    The following `BAOAB integration scheme <https://doi.org/10.1007/978-3-319-16375-8>`_ is used, where::

        B = deterministic velocity update
        A = deterministic position update
        O = stochastic velocity update

    We have implemented the following update so as to only calculate
    forces once per timestep:

    .. math::
        [B]\;& V_{t+1/2} = V_t + \frac{\Delta t}{2m}  F(X_t) \\
        [A]\;& X_{t+1/2} = X_t + \frac{\Delta t}{2}V_{t+1/2}  \\
        [O]\;& \tilde{V}_{t+1/2} = \epsilon V_{t+1/2} + \alpha dW_t \\
        [A]\;& X_{t+1} = X_{t+1/2} + \frac{\Delta t}{2} \tilde{V}_{t+1/2}  \\
        [B]\;& V_{t+1} = \tilde{V}_{t+1/2} + \frac{\Delta t}{2m}  F(X_{t+1})

    Where, :math:`dW_t` is a noise drawn from :math:`\mathcal{N}(0,1)`,
    :math:`\eta` is the friction, :math:`\epsilon` is the velocity scale,
    :math:`\alpha` is the noise scale, and:

    .. math::
        F(X_t) =& - \nabla  U(X_t)  \\
        \epsilon =& \exp(-\eta \; \Delta t) \\
        \alpha =& \sqrt{1 - \epsilon^2}

    A diffusion constant :math:`D` can be back-calculated using
    the Einstein relation:

    .. math::
        D = 1 / (\beta  \eta)

    Initial velocities are set to zero if not provided.

    Parameters
    ----------

    friction :
        Friction value for Langevin scheme, in units of inverse time.

    """

    def __init__(self, friction: float = 1e-3, **kwargs: Any):

        super(LangevinSimulation, self).__init__(**kwargs)

        assert friction > 0
        self.friction = friction

        self.vscale = np.exp(-self.dt * self.friction)
        self.noisescale = np.sqrt(1 - self.vscale * self.vscale)

    def timestep(
        self, data: AtomicData, forces: torch.Tensor
    ) -> Tuple[AtomicData, torch.Tensor, torch.Tensor]:
        """Timestep method for Langevin dynamics
        Parameters
        ----------
        data:
            atomic structure at t
        forces:
            forces evaluated at t
        Returns
        -------
        data:
            atomic structure at t+1
        forces:
            forces evaluated at t+1
        potential:
            potential evaluated at t+1
        """
        v_old = data[VELOCITY_KEY]
        masses = data.masses
        x_old = data[POSITIONS_KEY]

        # B
        v_new = v_old + 0.5 * self.dt * forces / masses[:, None]

        # A (position update)
        x_new = x_old + v_new * self.dt * 0.5

        # O (noise)
        noise = torch.sqrt(1.0 / self.beta / masses[:, None])
        noise = noise * torch.randn(
            size=x_new.size(),
            dtype=x_new.dtype,
            generator=self.rng,
            device=self.device,
        )
        v_new = v_new * self.vscale + self.noisescale * noise

        # A
        x_new = x_new + v_new * self.dt * 0.5

        data[POSITIONS_KEY] = x_new
        potential, forces = self.calculate_potential_and_forces(data)
        # B
        v_new = v_new + 0.5 * self.dt * forces / masses[:, None]
        data[VELOCITY_KEY] = v_new

        return data, potential, forces

    def attach_configurations(self, configurations: List[AtomicData]):
        """Setup the starting atomic configurations.

        Parameters
        ----------
        configurations : List[AtomicData]
            List of AtomicData instances representing initial structures for
        parallel simulations.
        """
        super().attach_configurations(configurations)

        if VELOCITY_KEY not in self.initial_data:
            # initialize velocities at zero
            self.initial_data[VELOCITY_KEY] = torch.zeros_like(
                self.initial_data[POSITIONS_KEY]
            )

        assert (
            self.initial_data[VELOCITY_KEY].shape
            == self.initial_data[POSITIONS_KEY].shape
        )

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
        potential: torch.Tensor,
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

        v_new = data[VELOCITY_KEY].view(-1, self.n_atoms, self.n_dims)
        masses = data.masses.view(self.n_sims, self.n_atoms)

        save_ind = t // self.save_interval

        if self.save_energies:
            kes = 0.5 * torch.sum(
                torch.sum(masses[:, :, None] * v_new**2, dim=2), dim=1
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


# pipe the doc from the base class into the child class so that it's properly
# displayed by sphinx
LangevinSimulation.__doc__ += _Simulation.__doc__


class OverdampedSimulation(_Simulation):
    r"""Overdamped Langevin simulation class for trained models.

    The following Brownian motion scheme is used:

    .. math::

        dX_t = - \nabla U( X_t )   D  \Delta t + \sqrt{( 2  D *\Delta t / \beta )} dW_t

    for coordinates :math:`X_t` at time :math:`t`, potential energy :math:`U`,
    diffusion :math:`D`, thermodynamic inverse temperature :math:`\beta`,
    time step :math:`\Delta t`, and stochastic Weiner process :math:`W`.

    Parameters
    ----------
    diffusion :
        The constant diffusion parameter :math:`D`. By default, the diffusion
        is set to unity and is absorbed into the :math:`\Delta t` argument.
        However, users may specify separate diffusion and :math:`\Delta t`
        parameters in the case that they have some estimate of the diffusion.
    """

    def __init__(self, diffusion: float = 1.0, **kwargs: Any):

        super(OverdampedSimulation, self).__init__(**kwargs)

        assert diffusion is not None
        assert diffusion > 0
        self.diffusion = diffusion
        self._dtau = self.diffusion * self.dt

    def attach_configurations(self, configurations: List[AtomicData]):
        """Setup the starting atomic configurations.

        Parameters
        ----------
        configurations : List[AtomicData]
            List of AtomicData instances representing initial structures for
        parallel simulations.
        """
        super().attach_configurations(configurations)
        if MASS_KEY in self.initial_data:
            warnings.warn(
                "Masses were provided, but will not be used since "
                "an overdamped Langevin scheme is being used for integration."
            )

    def timestep(
        self, data: AtomicData, forces: torch.Tensor
    ) -> Tuple[AtomicData, torch.Tensor, torch.Tensor]:
        """Timestep method for overdamped Langevin dynamics
        Parameters
        ----------
        data:
            atomic structure at t
        forces:
            forces evaluated at t
        Returns
        -------
        data:
            atomic structure at t+1
        forces:
            forces evaluated at t+1
        potential:
            potential evaluated at t+1
        """
        x_old = data[POSITIONS_KEY]
        noise = torch.randn(size=x_old.size(), generator=self.rng).to(
            self.device
        )
        x_new = (
            x_old.detach()
            + forces * self._dtau
            + np.sqrt(2 * self._dtau / self.beta) * noise
        )
        data[POSITIONS_KEY] = x_new
        potential, forces = self.calculate_potential_and_forces(data)
        return data, potential, forces


# pipe the doc from the base class into the child class so that it's properly
# displayed by sphinx
OverdampedSimulation.__doc__ += _Simulation.__doc__


class PTSimulation(LangevinSimulation):
    """Parallel tempering simulation using a Langevin update scheme.
    For thoeretical details on replica exchange/parallel tempering, see
    https://github.com/noegroup/reform.
    Note that currently we only implement parallel tempering for Langevin dynamics.
    Be aware that the output will contain information (e.g., coordinates)
    for all replicas.

    Parameters
    ----------
    friction:
        Scalar friction to use for Langevin updates
    betas:
        List of inverse temperatures for each of the thermodynamic replicas
    exchange_interval:
        Specifies teh number of simulation steps to take before attempting
        replica exchange.
    """

    def __init__(
        self,
        friction: float = 1e-3,
        betas: List[float] = [1.67, 1.42, 1.32],
        exchange_interval: int = 100,
        **kwargs,
    ):

        assert all([beta > 0.00 for beta in betas])
        self.betas = betas
        self.n_replicas = len(self.betas)

        super(PTSimulation, self).__init__(
            friction=friction,
            specific_setup=self._reset_exchange_stats,
            subroutine=self.detect_and_exchange_replicas,
            subroutine_interval=exchange_interval,
            **kwargs,
        )
        self._reset_exchange_stats()

    def _reset_exchange_stats(self):
        """Setup function that resets exchange statistics before running a simulation"""
        self._replica_exchange_attempts = 0
        self._replica_exchange_approved = 0

    def timestep(
        self, data: AtomicData, forces: torch.Tensor
    ) -> Tuple[AtomicData, torch.Tensor, torch.Tensor]:
        """Timestep method for Langevin dynamics
        Parameters
        ----------
        data:
            atomic structure at t
        forces:
            forces evaluated at t
        Returns
        -------
        data:
            atomic structure at t+1
        forces:
            forces evaluated at t+1
        potential:
            potential evaluated at t+1
        """
        v_old = data[VELOCITY_KEY]
        masses = data.masses
        x_old = data[POSITIONS_KEY]

        # B
        v_new = v_old + 0.5 * self.dt * forces / masses[:, None]

        # A (position update)
        x_new = x_old + v_new * self.dt * 0.5

        # O (noise)
        noise = torch.sqrt(1.0 / self.beta / masses[:, None])
        noise = noise * torch.randn(
            size=x_new.size(),
            dtype=x_new.dtype,
            generator=self.rng,
            device=self.device,
        )
        v_new = v_new * self.vscale + self.noisescale * noise

        # A
        x_new = x_new + v_new * self.dt * 0.5

        data[POSITIONS_KEY] = x_new
        potential, forces = self.calculate_potential_and_forces(data)
        # B
        v_new = v_new + 0.5 * self.dt * forces / masses[:, None]
        data[VELOCITY_KEY] = v_new

        return data, potential, forces

    def attach_configurations(self, configurations: List[AtomicData]):
        """Attaches the configurations at each of the temperatures defined for
        parallel tempering simulations"""
        # copy the configurations across each beta/temperature
        new_configurations = []
        for beta in self.betas:
            for configuration in configurations:
                config = deepcopy(configuration)
                config[BETA_KEY] = torch.tensor(beta)
                new_configurations.append(config)

        # collate the final datalist
        self.validate_data_list(new_configurations)
        self.initial_data = self.collate(new_configurations).to(
            device=self.device
        )
        # If not provided, initialize all velocites to zero
        if VELOCITY_KEY not in self.initial_data:
            # initialize velocities at zero
            self.initial_data.velocities = torch.zeros_like(
                self.initial_data.pos
            )
        assert (
            self.initial_data[VELOCITY_KEY].shape
            == self.initial_data[POSITIONS_KEY].shape
        )

        self.n_sims = len(new_configurations)
        self.n_indep_sims = len(configurations)
        self.n_atoms = len(new_configurations[0].atom_types)
        self.n_dims = new_configurations[0].pos.shape[1]

        # Setup possible even/odd pair exchanges
        self._propose_even_pairs = True

        # (0, 1), (2, 3), ...
        even_pairs = [(i, i + 1) for i in torch.arange(self.n_replicas)[:-1:2]]
        # (1, 2), (3, 4), ...
        odd_pairs = [(i, i + 1) for i in torch.arange(self.n_replicas)[1:-1:2]]
        if len(odd_pairs) == 0:
            odd_pairs = even_pairs
        pair_a = []
        pair_b = []
        for pair in even_pairs:
            pair_a.append(
                torch.arange(self.n_indep_sims) + pair[0] * self.n_indep_sims
            )
            pair_b.append(
                torch.arange(self.n_indep_sims) + pair[1] * self.n_indep_sims
            )
        self._even_pairs = [np.concatenate(pair_a), np.concatenate(pair_b)]
        pair_a = []
        pair_b = []
        for pair in odd_pairs:
            pair_a.append(
                torch.arange(self.n_indep_sims) + pair[0] * self.n_indep_sims
            )
            pair_b.append(
                torch.arange(self.n_indep_sims) + pair[1] * self.n_indep_sims
            )
        self._odd_pairs = [torch.cat(pair_a), torch.cat(pair_b)]

    def get_replica_info(self, replica_num: int = 0) -> Dict:
        """Returns information for the specified replica after the
        parallel tempering simulation has completed

        Parameters
        ----------
        replica_num:
            integer specifying which replica to interrogate

        Returns
        -------
        dict:
            dictionary with replica exchange information about the
            desired replica
        """

        if (
            type(replica_num) is not int
            or replica_num < 0
            or replica_num >= self.n_replicas
        ):
            raise ValueError("Please provide a valid replica number.")
        indices = torch.arange(
            replica_num * self.n_indep_sims,
            (replica_num + 1) * self.n_indep_sims,
        )
        return {
            "beta": self._betas[replica_num],
            "indices_in_the_output": indices,
        }

    def _get_proposed_pairs(self) -> List[torch.Tensor]:
        """Proposes the even and odd exchange pairs alternatively.

        Returns
        -------
        torch.Tensor:
            pair_a, all possible even/odd pairs across all of the simulations/replicas
        torch.Tensor:
            pair_b, all possible
        """
        if self._propose_even_pairs:
            self._propose_even_pairs = False
            return self._even_pairs
        else:
            self._propose_even_pairs = True
            return self._odd_pairs

    def _detect_exchange(self, data: AtomicData) -> Dict:
        """Proposes and checks pairs to be exchanged for parallel tempering.
        Modified from `reform`. Briefly, a pair excahnge is proposed, and the
        current associated potential energies are used to compute a Boltzmann ratio based
        on the temperature/energy differences. This ratio defines an acceptance threshold
        aginst which approved exchanges are sampled according to a normal distribution.

        Parameters
        ----------
        data:
            Collated AtomicData instance containing the beta values and current potential
            energies for each replica.

        Returns
        -------
        dict:
            Dictionary containing the approved exchanges
        """
        pair_a, pair_b = self._get_proposed_pairs()
        u_a, u_b = data.out[ENERGY_KEY][pair_a], data.out[ENERGY_KEY][pair_b]
        betas_a, betas_b = data[BETA_KEY][pair_a], data[BETA_KEY][pair_b]

        p_pair = torch.exp((u_a - u_b) * (betas_a - betas_b))
        approved = torch.rand(len(p_pair)) < p_pair
        self._replica_exchange_attempts += len(pair_a)
        self._replica_exchange_approved += torch.sum(approved).numpy()
        pairs_for_exchange = {"a": pair_a[approved], "b": pair_b[approved]}
        return pairs_for_exchange

    def _perform_exchange(
        self, data: AtomicData, pairs_for_exchange: Dict
    ) -> AtomicData:
        """Exchanges the coordinates and velcities for those pairs marked for exchange

        Parameters
        ----------
        data:
            Collated AtomicData instance containing the current Cartesian coordinates and
            velocities for each simulation/replica
        pairs_for_exchange:
            Dictionary that denotes which pairs have been accepted for exchange

        Returns
        -------
        AtomicData:
            The updated collated atomic data where the coordinates and (rescaled) velocities
            have been exchanged according to the appropriate supplied exchange pairs
        """
        pair_a, pair_b = pairs_for_exchange["a"], pairs_for_exchange["b"]
        # exchange the coordinates
        # Here we must make swaps in the coordinates and velocities
        # according to to the collated batch attribute

        batch_pair_a_cond = False
        batch_pair_b_cond = False
        for idx_a, idx_b in zip(pair_a, pair_b):
            # cumulative bitwise OR to grab corresponding pair batch index
            batch_pair_a_cond |= data.batch == idx_a
            batch_pair_b_cond |= data.batch == idx_b

        # exchange coordinates
        x_changed = data[POSITIONS_KEY].detach().clone()
        x_changed[batch_pair_a_cond] = data[POSITIONS_KEY][batch_pair_b_cond]
        x_changed[batch_pair_b_cond] = data[POSITIONS_KEY][batch_pair_a_cond]

        # scale and exchange the velocities
        # reshape the betas for simpler elementwise multiplication
        betas_a = data[BETA_KEY][pair_a].repeat_interleave(self.n_atoms)[
            :, None
        ]
        betas_b = data[BETA_KEY][pair_b].repeat_interleave(self.n_atoms)[
            :, None
        ]

        vscale_a_to_b = torch.sqrt(betas_b / betas_a)
        vscale_b_to_a = torch.sqrt(betas_a / betas_b)
        v_changed = data[VELOCITY_KEY].detach().clone()
        v_changed[batch_pair_a_cond] = (
            data[VELOCITY_KEY][batch_pair_b_cond] * vscale_a_to_b
        )
        v_changed[batch_pair_b_cond] = (
            data[VELOCITY_KEY][batch_pair_a_cond] * vscale_b_to_a
        )

        data[POSITIONS_KEY] = x_changed
        data[VELOCITY_KEY] = v_changed
        return data

    def detect_and_exchange_replicas(self, data: AtomicData) -> AtomicData:
        """Subroutine for replica exchange: Modifies the internal coordinates and velocities
        according to the algorithm specified by `reform`:

        https://github.com/noegroup/reform

        Parameters
        ----------
        data:
            Current `AtomicData` instance containing all replicas, their coordinates, velocities,
            potential energies, and beta values

        Returns
        -------
        data:
            Updated `AtomicData` instance containing potentially exchanged replicas.
        """
        pairs_for_exchange = self._detect_exchange(data)
        data = self._perform_exchange(data, pairs_for_exchange)
        return data

    def summary(self):
        attempted = self._replica_exchange_attempts
        exchanged = self._replica_exchange_approved
        printstring = "Done simulating ({})".format(time.asctime())
        printstring += "\nReplica-exchange rate: %.2f%% (%d/%d)" % (
            exchanged / attempted * 100.0,
            exchanged,
            attempted,
        )
        printstring += (
            "\nNote that you can call .get_replica_info"
            "(#replica) to query the inverse temperature"
            " and trajectory indices for a given replica."
        )
        if self.log_type == "print":
            print(printstring)
        elif self.log_type == "write":
            printstring += "\n"
            with open(self._log_file, "a") as lfile:
                lfile.write(printstring)


# pipe the doc from the base class into the child class so that it's properly
# displayed by sphinx
PTSimulation.__doc__ += _Simulation.__doc__
