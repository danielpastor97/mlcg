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
from .langevin import LangevinSimulation


class PTSimulation(LangevinSimulation):
    """Parallel tempering simulation using a Langevin update scheme.
    For theoretical details on replica exchange/parallel tempering, see
    https://github.com/noegroup/reform.
    Note that currently we only implement parallel tempering for Langevin dynamics.
    Be aware that the output will contain information (e.g., coordinates)
    for all replicas.

    Note: This implementation only allows for replica exchanges between directly
    adjecent temperatures implied by the user-supplied list of beta values.

    Parameters
    ----------
    friction:
        Scalar friction to use for Langevin updates
    betas:
        List of inverse temperatures for each of the thermodynamic replicas
    exchange_interval:
        Specifies the number of simulation steps to take before attempting
        replica exchange.
    """

    def __init__(
        self,
        friction: float = 1e-3,
        betas: List[float] = [1.67, 1.42, 1.17],
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
        """Timestep method for Langevin dynamics. Here, the scale
        for the noise term is conditioned on the beta values for each respective replica.

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
        noise = torch.sqrt(
            1.0 / data[BETA_KEY].repeat_interleave(self.n_atoms) / masses
        )[:, None]
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
        parallel tempering simulations. If the initial configurations do not contain
        specified velocities, all velocities will be initialized to zero.
        """
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
            "beta": self.betas[replica_num],
            "indices_in_the_output": indices,
        }

    def _get_proposed_pairs(self) -> List[torch.Tensor]:
        """Proposes the even and odd exchange pairs alternatively each time the
        _detect_exchange method is called. Exchanges can only happen between direcly adjacent
        temperatures defined by the user supplied beta series.

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
        Modified from `reform`. Briefly, a pair exchange is proposed, and the
        current associated potential energies are used to compute a Boltzmann ratio based
        on the temperature/energy differences. This ratio defines an acceptance threshold
        aginst which approved exchanges are sampled according to a unit uniform distribution.
        For a pair of configurations :math:`A` and :math:`B`, characterized by the respective
        potential energies :math:`U_A` and :math:`U_B` the the inverse thermodynamic temperatures
        :math:`\beta_A` and :math:`\beta_B`, the acceptance rate for exchanging configurations is:

        .. math::

            Acc = \exp{\left( (U_A - U_B) \times (\beta_A - \beta_B) \right)}

        Pairs of candidate configurations undergo exchange if :math:`\rho \sim U(0,1) < Acc`. Note
        that the exchanged velocities for each configuration must further be rescaled according to the
        square root of their inverse beta ratios. See _perform_exchange for more details.

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
        """Exchanges the coordinates and velcities for those pairs marked for exchange.
        Exchanged velocities are rescaled based on ratios of beta values from the two configurations.
        For a pair of configurations :math:`A` and :math:`B`, characterized by the respective
        potential energies :math:`U_A` and :math:`U_B` the the inverse thermodynamic temperatures
        :math:`\beta_A` and :math:`\beta_B`, the the velocity exchange rescaling factor is:

        .. math::

            vscale = \sqrt{\frac{\beta_{old}}{\beta_{new}}}

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

        vscale_a_to_b = torch.sqrt(betas_a / betas_b)
        vscale_b_to_a = torch.sqrt(betas_b / betas_a)
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
