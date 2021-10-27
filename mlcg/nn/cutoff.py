import torch
import torch.nn as nn
import numpy as np
import math


class _Cutoff(nn.Module):
    r"""Abstract cutoff class"""

    def __init__(self):
        super(_Cutoff, self).__init__()
        self.cutoff_lower = None
        self.cutoff_upper = None

    def check_cutoff(self):
        if self.cutoff_upper < self.cutoff_lower:
            raise ValueError("Upper cutoff is less than lower cutoff")

    def forward(self):
        raise NotImplementedError


class _OneSidedCutoff(nn.Module):
    r"""Abstract classs for cutoff functions with a fuxed lower cutoff of 0"""

    def __init__(self):
        super(_OneSidedCutoff, self).__init__()
        self.cutoff_lower = 0
        self.cutoff_upper = None

    def forward(self):
        raise NotImplementedError


class IdentityCutoff(_Cutoff):
    r"""Cutoff function that applies an identity transform, but retains
    cutoff_lower and cutoff_upper attributes

    Parameters
    ----------
    cutoff_lower:
        left bound for the radial cutoff distance
    cutoff_upper:
        right bound for the radial cutoff distance
    """

    def __init__(self, cutoff_lower: float = 0, cutoff_upper: float = np.inf):
        super(IdentityCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

        self.check_cutoff()

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        r"""Applies identity transform to input distances

        Parameters
        ----------
        dist:
            Input distances of shape (total_num_distances)

        Returns
        -------
            Identity-transformed output distances of shape (total_num_edges)
        """
        return dist


class CosineCutoff(_Cutoff):
    r"""Class implementing a cutoff envelope based a cosine signal in the
    interval `[lower_cutoff, upper_cutoff]`:

    .. math::

        \cos{\left( r_{ij} \times \pi / r_{high})\right)} + 1.0

    NOTE: The behavior of the cutoff is qualitatively different for lower
    cutoff values greater than zero when compared to the zero lower cutoff
    default. We recommend visualizing your basis to see if it makes physical
    sense.

    .. math::

        0.5 \times ( \cos{ ( \pi (2 \frac{r_{ij} - r_{low}}{r_{high}
         - r_{low}} + 1.0))} + 1.0 )

    """

    def __init__(self, cutoff_lower: float = 0.0, cutoff_upper: float = 5.0):
        super(CosineCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

        self.check_cutoff()

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """Applies cutoff envelope to distances.

        Parameters
        ----------
        distances:
            Distances of shape (total_num_edges)

        Returns
        -------
        cutoffs:
            Distances multiplied by the cutoff envelope, with shape
            (total_num_edges)
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
