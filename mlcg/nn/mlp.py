import torch
from torch import nn
from typing import List

class MLP(nn.Module):
  '''
    Multilayer Perceptron for regression of scalars
  '''
  def __init__(self, layer_widths: List[int] = [10, 10, 1],
                        activation_func: nn.Module = torch.nn.Tanh()):
    super(MLP, self).__init__()
    layers = []
    for w_in, w_out in zip(layer_widths[:-2], layer_widths[1:-1]):
        layers.append(nn.Linear(w_in, w_out, bias=True, dtype=torch.float64))
        layers.append(activation_func)
    # last layer without activation function and bias
    layers.append(nn.Linear(layer_widths[-2], layer_widths[-1], bias=False, dtype=torch.float64))

    self.layers = nn.Sequential(*layers)


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)