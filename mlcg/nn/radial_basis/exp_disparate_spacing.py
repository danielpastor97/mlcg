import torch
from torch import nn
from typing import Union

from .base import _RadialBasis
from ..cutoff import _Cutoff, CosineCutoff


class SpacedNormalBasis(_RadialBasis):
    r"""Class for generating a set of exponential normal radial basis functions,
    as described in [Physnet]_ . The functions have the following form:

    .. math::

        
    where

    .. math::

        
    is a distance rescaling factor, and, by default

    .. math::


    represents a cosine cutoff function (though users can specify their own cutoff function
    if they desire).

    Parameters
    ----------
    cutoff:
        Defines the smooth cutoff function. If a float is provided, it will be interpreted as
        an upper cutoff and a CosineCutoff will be used between 0 and the provided float. Otherwise,
        a chosen `_Cutoff` instance can be supplied.
    num_rbf:
        The number of functions in the basis set.
    trainable:
        If True, the parameters of the basis (the centers and widths of each
        function) are registered as optimizable parameters that will be updated
        during backpropagation. If False, these parameters will be instead fixed
        in an unoptimizable buffer.
    """

    def __init__(
        self,
        cutoff: Union[int, float, _Cutoff],
        num_rbf: int = 50,
        sigma_min: float = 0.25,
        sigma_factor: float = 2.0,
        spacing: float = 2.0,
        trainable: bool = True,
    ):
        super(SpacedNormalBasis, self).__init__()
        if isinstance(cutoff, (float, int)):
            self.cutoff = CosineCutoff(0, cutoff)
        elif isinstance(cutoff, _Cutoff):
            self.cutoff = cutoff
        else:
            raise TypeError(
                "Supplied cutoff {} is neither a number nor a _Cutoff instance.".format(
                    cutoff
                )
            )

        self.num_rbf = num_rbf
        self.sigma_min = sigma_min
        self.sigma_factor = sigma_factor
        self.spacing = spacing

        self.check_cutoff()

        self.num_rbf = num_rbf
        self.trainable = trainable

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        r"""
        Method for initializing the basis function parameters
        """

        mus = [0,self.sigma_factor]
        sigmas = [self.sigma_factor/self.spacing, self.sigma_min]
        while mus[-1] < self.cutoff_upper:
            mus.append(mus[-1]+self.spacing*sigmas[-1])
            sigmas.append(self.sigma_factor*sigmas[-1])
        means = torch.FloatTensor(mus)
        betas = torch.FloatTensor(sigmas)
        return means, betas

    def reset_parameters(self):
        r"""
        Method to reset the parameters of the basis functions to their
        initial values.
        """
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        r"""
        Expansion of distances through the radial basis function set.

        Parameters
        ----------
        dist: torch.Tensor
            Input pairwise distances of shape (total_num_edges)

        Return
        ------
        expanded_distances: torch.Tensor
            Distances expanded in the radial basis with shape (total_num_edges, num_rbf)
        """

        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist)*torch.exp(-(dist-self.means)**2/(2*self.betas**2))
