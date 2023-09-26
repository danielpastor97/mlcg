import torch
from typing import List
from ._module_init import init_xavier_uniform


class MLP(torch.nn.Module):
    """Multilayer Perceptron for regression.

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
    last_bias: bool
        add a bias term to the last layer of the MLP
    """

    def __init__(
        self,
        layer_widths: List[int] = None,
        activation_func: torch.nn.Module = torch.torch.nn.Tanh(),
        last_bias: bool = True,
    ):
        super(MLP, self).__init__()
        if layer_widths is None:
            layer_widths = [10, 10, 1]
        layers = []
        for w_in, w_out in zip(layer_widths[:-2], layer_widths[1:-1]):
            layers.append(torch.nn.Linear(w_in, w_out, bias=True))
            layers.append(activation_func)
        # last layer without activation function and bias
        layers.append(
            torch.nn.Linear(layer_widths[-2], layer_widths[-1], bias=last_bias)
        )

        self.layers = torch.nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            init_xavier_uniform(layer)

    def forward(self, x, data=None):
        """Forward pass"""
        return self.layers(x)
