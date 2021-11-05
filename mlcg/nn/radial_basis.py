import torch
from torch import nn
from .cutoff import CosineCutoff


def visualize_basis(rbf_layer: nn.Module):
    r"""Function for quickly visualizing a specific basis. This is useful for
    inspecting the distance coverage of basis functions for non-default lower
    and upper cutoffs.

    Parameters
    ----------
    rbf_layer:
        Input radial basis function layer to visualize.
    """

    import matplotlib.pyplot as plt

    distances = torch.linspace(
        rbf_layer.cutoff_lower - 1, rbf_layer.cutoff_upper + 1, 1000
    )
    expanded_distances = rbf_layer(distances)

    for i in range(expanded_distances.shape[-1]):
        plt.plot(distances.numpy(), expanded_distances[:, i].detach().numpy())
    plt.show()


class _RadialBasis(nn.Module):
    r"""Abstract radial basis function class"""

    def __init__(self):
        super(_RadialBasis, self).__init__()
        self.cutoff_lower = None
        self.cutoff_upper = None

    def check_cutoff(self):
        if self.cutoff_upper < self.cutoff_lower:
            raise ValueError(
                "Upper cutoff {} is less than lower cutoff {}".format(
                    self.cutoff_upper, self.cutoff_lower
                )
            )

    def forward(self):
        raise NotImplementedError


class GaussianBasis(_RadialBasis):
    r"""Class that generates a set of equidistant 1-D gaussian basis functions
    scattered between a specified lower and upper cutoff:

    .. math::

        f_n = \exp{ \left( -\gamma(r-c_n)^2 \right) }

    Parameters
    ----------
    cutoff_lower:
        Lower distance cutoff, corresponding to the center of the first gaussian
        function in the basis.
    cutoff_upper:
        Upper distance cutoff, corresponding to the center of the last gaussian
        function in the basis.
    num_rbf:
        The number of gaussian functions in the basis set.
    trainable:
        If True, the parameters of the gaussian basis (the centers and widths of
        each function) are registered as optimizable parameters that will be
        updated during backpropagation. If False, these parameters will be
        instead fixed in an unoptimizable buffer.

    """

    def __init__(
        self,
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 5.0,
        num_rbf: int = 50,
        trainable: bool = False,
    ):
        super(GaussianBasis, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

        self.check_cutoff()

        self.num_rbf = num_rbf
        self.trainable = trainable

        offset, coeff = self._initial_params()
        if trainable:
            self.register_parameter("coeff", nn.Parameter(coeff))
            self.register_parameter("offset", nn.Parameter(offset))
        else:
            self.register_buffer("coeff", coeff)
            self.register_buffer("offset", offset)

    def _initial_params(self):
        r"""Method for generating the initial parameters of the basis.
        The functions are set to have equidistant centers between the
        lower and cupper cutoff, while the variance of each function
        is set based on the difference between the lower and upper
        cutoffs.
        """
        offset = torch.linspace(
            self.cutoff_lower, self.cutoff_upper, self.num_rbf
        )
        coeff = -0.5 / (offset[1] - offset[0]) ** 2
        return offset, coeff

    def reset_parameters(self):
        r"""Method for resetting the basis to its initial state"""
        offset, coeff = self._initial_params()
        self.offset.data.copy_(offset)
        self.coeff.data.copy_(coeff)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        r"""Expansion of distances through the radial basis function set.

        Parameters
        ----------
        dist:
            Input pairwise distances of shape (total_num_edges)

        Return
        ------
        expanded_distances:
            Distances expanded in the radial basis with shape (total_num_edges,
            num_rbf)
        """

        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ExpNormalBasis(_RadialBasis):
    r"""Class for generating a set of exponential normal radial basis functions,
    as described in [Physnet]_ . The functions have the following form:

    .. math::

        f_n(r_{ij};\alpha, r_{low},r_{high}) = f_{cut}(r_{ij},r_{low},r_{high})
        \times \exp\left[-\beta_n \left(e^{\alpha (r_{ij} -r_{high}) }
        - \mu_n \right)^2\right]

    where

    .. math::

        \alpha = 5.0/(r_{high} - r_{low})

    is a distance rescaling factor, and

    .. math::

        f_{cut} ( r_{ij},r_{low},r_{high} ) =  \cos{\left( r_{ij} \times \pi / r_{high}\right)} + 1.0

    Parameters
    ----------
    cutoff_lower:
        Lower distance cutoff, corresponding to the center of the first gaussian
        function in the basis.
    cutoff_upper:
        Upper distance cutoff, corresponding to the zero point off the cutoff
        envelope.
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
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 5.0,
        num_rbf: int = 50,
        trainable: bool = True,
    ):
        super(ExpNormalBasis, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

        self.check_cutoff()

        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff = CosineCutoff(0, cutoff_upper)
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        r"""Method for initializing the basis function parameters, as described in
        https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181 .
        """

        start_value = torch.exp(
            torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower)
        )
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def reset_parameters(self):
        r"""Method to reset the parameters of the basis functions to their
        initial values.
        """
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        r"""Expansion of distances through the radial basis function set.

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
        return self.cutoff(dist) * torch.exp(
            -self.betas
            * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means)
            ** 2
        )
