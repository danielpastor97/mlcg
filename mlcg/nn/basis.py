import math
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


def visualize_basis(rbf_layer):
    """Function for quickly visualizing a specific basis. This is useful for inspecting
    the distance coverage of basis functions for non-default lower and upper cutoffs.

    Parameters
    ----------
    rbf_layer: nn.Module
        Input radial basis function layer to visualize.
    """

    import matplotlib.pyplot as plt

    distances = torch.linspace(cutoff_lower - 1, cutoff_upper + 1, 1000)
    expanded_distances = rbf_layer(distances)

    for i in range(expanded_distances.shape[-1]):
        plt.plot(distances.numpy(), expanded_distances[:, i].detach().numpy())
    plt.show()


class GaussianBasis(nn.Module):
    """Class that generates a set of equidistant 1-D gaussian basis functions
    scattered between a specified lower and upper cutoff.

    Parameters
    ----------
    cutoff_lower: float (default=0.0)
        Lower distance cutoff, corresponding to the center of the first gaussian
        function in the basis.
    cutoff_upper: float (default=5.0)
        Upper distance cutoff, corresponding to the center of the last gaussian
        function in the basis.
    num_rbf: int (default=50)
        The number of gaussian functions in the basis set.
    trainable: bool (default=False)
        If True, the parameters of the gaussian basis (the centers and widths of each
        function) are registered as optimizable parameters that will be updated during
        backpropagation. If False, these parameters will be instead fixed in an
        unoptimizable buffer.
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
        """Method for generating the initial parameters of the basis. The functions
        are set to have equidistant centers between the lower and cupper cutoff, while
        the variance of each function is set based on the difference between the lower
        and upper cutoffs.
        """
        offset = torch.linspace(
            self.cutoff_lower, self.cutoff_upper, self.num_rbf
        )
        coeff = -0.5 / (offset[1] - offset[0]) ** 2
        return offset, coeff

    def reset_parameters(self):
        """Method for resetting the basis to its initial state"""
        offset, coeff = self._initial_params()
        self.offset.data.copy_(offset)
        self.coeff.data.copy_(coeff)

    def forward(self, dist):
        """Expansion of distances through the radial basis function set.

        Parameters
        ----------
        dist: torch.Tensor
            Input pairwise distances of shape (total_num_edges)

        Return
        ------
        expanded_distances: torch.Tensor
            Distances expanded in the radial basis with shape (total_num_edges, num_rbf)
        """

        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ExpNormalBasis(nn.Module):
    """Class for generating a set of exponential normal radial basis functions, as described in
    the following paper:

    Unke, O. T., & Meuwly, M. (2019). PhysNet: A Neural Network for Predicting Energies, Forces, Dipole Moments and Partial Charges. Journal of Chemical Theory and Computation, 15(6), 3678–3693. https://doi.org/10.1021/acs.jctc.9b00181

    Parameters
    ----------
    cutoff_lower: float (default=0.0)
        Lower distance cutoff, corresponding to the center of the first gaussian
        function in the basis.
    cutoff_upper: float (default=5.0)
        Upper distance cutoff, corresponding to the zero point off the cutoff envelope.
    num_rbf: int (default=50)
        The number of functions in the basis set.
    trainable: bool (default=False)
        If True, the parameters of the basis (the centers and widths of each
        function) are registered as optimizable parameters that will be updated during
        backpropagation. If False, these parameters will be instead fixed in an
        unoptimizable buffer.
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
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(0, cutoff_upper)
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        """Method for initializing the basis function parameters, as described in
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
        """Method to reset the parameters of the basis functions to their initial values."""
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        """Expansion of distances through the radial basis function set.

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
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas
            * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means)
            ** 2
        )


class CosineCutoff(nn.Module):
    """Class implementing a cutoff envelope based a cosine signal.

    NOTE: The behavior of the cutoff is qualitatively different for lower
    cutoff values greater than zero when compared to the zero lower cutoff default.
    We recommend visualizing your basis to see if it makes physical sense.
    """

    def __init__(self, cutoff_lower: float = 0.0, cutoff_upper: float = 5.0):
        super(CosineCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances):
        """Applies cutoff envelope to distances.

        Parameters
        ----------
        distances: torch.Tensor
            Distances of shape (total_num_edges)

        Returns
        -------
        cutoffs: torch.Tensor
            Distances multiplied by the cutoff envelope, with shape (total_num_edges)
        """
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                torch.cos(
                    math.pi
                    * (
                        2
                        * (distances - self.cutoff_lower)
                        / (self.cutoff_upper - self.cutoff_lower)
                        + 1.0
                    )
                )
                + 1.0
            )
            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            cutoffs = cutoffs * (distances > self.cutoff_lower).float()
            return cutoffs
        else:
            cutoffs = 0.5 * (
                torch.cos(distances * math.pi / self.cutoff_upper) + 1.0
            )
            # remove contributions beyond the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            return cutoffs
