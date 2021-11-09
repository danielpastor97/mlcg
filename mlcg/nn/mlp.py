import torch
from torch import nn
from typing import List


class MLP(nn.Module):
    """
    Multilayer perceptron for regression of scalars

    Note: No activation is applied to the final output,
    and the final output contains no learnable bias.

    Parameters
    ----------
    layer_widths:
        the width of outputs after passing through
        each linear layer. Eg, for an width specification:

        ..code-block::python

            [10, 10, 1]

        an n-dimensional example will be transformed
        to a 10-dimensional feature, then a second
        10-dimensional feature, then finally to a
        1-dimensional scalar feature.

    activation_func:
        The (non-linear) activation function to apply
        to the output of each linear transformation,
        with the exception of the final output, which
        is simply the output of a linear transformation
        with no bias
    """

    def __init__(
        self,
        layer_widths: List[int] = [10, 10, 1],
        activation_func: nn.Module = torch.nn.Tanh(),
    ):
        super(MLP, self).__init__()
        layers = []
        for w_in, w_out in zip(layer_widths[:-2], layer_widths[1:-1]):
            layers.append(nn.Linear(w_in, w_out, bias=True))
            layers.append(activation_func)
        # last layer without activation function and bias
        layers.append(
            nn.Linear(
                layer_widths[-2],
                layer_widths[-1],
                bias=False,
            )
        )

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass"""
        return self.layers(x)
