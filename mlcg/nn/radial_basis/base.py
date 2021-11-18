import torch


class _RadialBasis(torch.nn.Module):
    r"""Abstract radial basis function class"""

    def __init__(self):
        super(_RadialBasis, self).__init__()
        self.cutoff = None

    def check_cutoff(self):
        if self.cutoff.cutoff_upper < self.cutoff.cutoff_lower:
            raise ValueError(
                "Upper cutoff {} is less than lower cutoff {}".format(
                    self.cutoff.cutoff_upper, self.cutoff.cutoff_lower
                )
            )

    def forward(self):
        raise NotImplementedError