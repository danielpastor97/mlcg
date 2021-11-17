import torch
from typing import List


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
                bias=False,
            )
        )

        self.layers = torch.nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.layers[0].weight)
        self.layers[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.layers[2].weight)
        self.layers[2].bias.data.fill_(0)

    def forward(self, x):
        """Forward pass"""
        return self.layers(x)
