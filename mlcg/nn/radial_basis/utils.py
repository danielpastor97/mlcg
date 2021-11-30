import torch
from torch import nn


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
        rbf_layer.cutoff.cutoff_lower - 1,
        rbf_layer.cutoff.cutoff_upper + 1,
        1000,
    )
    expanded_distances = rbf_layer(distances)

    for i in range(expanded_distances.shape[-1]):
        plt.plot(distances.numpy(), expanded_distances[:, i].detach().numpy())
    plt.show()
