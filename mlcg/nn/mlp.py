import torch
from typing import List
from ._module_init import init_xavier_uniform

class MLP(torch.nn.Module):
    """
    Multilayer Perceptron for regression of scalars
    """

    def __init__(
        self,
        layer_widths: List[int] = [10, 10, 1],
        activation_func: torch.nn.Module = torch.torch.nn.Tanh(),
    ):
        super(MLP, self).__init__()
        layers = []
        for w_in, w_out in zip(layer_widths[:-2], layer_widths[1:-1]):
            layers.append(torch.nn.Linear(w_in, w_out, bias=True))
            layers.append(activation_func)
        # last layer without activation function and bias
        layers.append(
            torch.nn.Linear(
                layer_widths[-2],
                layer_widths[-1],
            )
        )

        self.layers = torch.nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            init_xavier_uniform(layer)


    def forward(self, x):
        """Forward pass"""
        return self.layers(x)
