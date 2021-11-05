from typing import List, Tuple, Any
import torch
import numpy as np
import warnings

from ..data.atomic_data import AtomicData
from ..data._keys import MASS_KEY, VELOCITY_KEY
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
        [O]\;& \tilde{V}_{t+1/2} = V_{t+1/2} \text{vscale} + dW_t  \text{noisescale} \\
        [A]\;& X_{t+1} = X_{t+1/2} + \frac{\Delta t}{2} \tilde{V}_{t+1/2}  \\
        [B]\;& V_{t+1} = \tilde{V}_{t+1/2} + \frac{\Delta t}{2m}  F(X_{t+1})

    Where, :math:`dW_t` is a noise drawn from :math:`\mathcal{N}(0,1)`, and:

    .. math::
        F(X_t) =& - \nabla  U(X_t)  \\
        \text{vscale} =& \exp[-\text{friction} \; \Delta t] \\
        \text{noisecale} =& \sqrt{1 - \text{vscale}^2}

    A diffusion constant :math:`D` can be back-calculated using
    the Einstein relation:

    .. math::
        D = 1 / (\beta  \text{friction})

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
        v_old = data.velocities
        masses = data.masses
        x_old = data.pos

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

        data.pos = x_new
        potential, forces = self.calculate_potential_and_forces(data)
        # B
        v_new = v_new + 0.5 * self.dt * forces / masses[:, None]
        data.velocities = v_new

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
            self.initial_data.velocities = torch.zeros_like(
                self.initial_data.pos
            )

        assert self.initial_data.velocities.shape == self.initial_data.pos.shape

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


# pipe the doc from the base class into the child class so that it's properly
# displayed by sphinx
LangevinSimulation.__doc__ += _Simulation.__doc__


class OverdampedSimulation(_Simulation):
    r"""Overdamped Langevin simulation class for trained models.

    The following Brownian motion scheme is used:

    .. math::

        dX_t = - \nabla U( X_t )   D  \Delta t + \sqrt( 2  D *\Delta t / \beta ) * dW_t

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
        potential, forces = self.calculate_potential_and_forces(data)
        return data, potential, forces


# pipe the doc from the base class into the child class so that it's properly
# displayed by sphinx
OverdampedSimulation.__doc__ += _Simulation.__doc__
