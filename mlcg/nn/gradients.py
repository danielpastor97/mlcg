import torch
from typing import Sequence
from ..data._keys import FORCE_KEY, ENERGY_KEY


class GradientsOut(torch.nn.Module):
    _targets = {FORCE_KEY: ENERGY_KEY}

    def __init__(self, model, targets=FORCE_KEY):
        super(GradientsOut, self).__init__()
        self.model = model
        self.name = self.model.name
        self.targets = []
        if isinstance(targets, str):
            self.targets = [targets]
        elif isinstance(targets, Sequence):
            self.targets = targets
        assert any(
            [k in GradientsOut._targets for k in self.targets]
        ), f"targets={self.targets} should be any of {GradientsOut._targets}"

    def forward(self, data):

        if self.name not in data.out:
            data.out[self.name] = {}

        if ENERGY_KEY not in data.out[self.name]:
            data.pos.requires_grad_(True)
            data = self.model(data)

        if FORCE_KEY in self.targets:
            y = data.out[self.name][ENERGY_KEY]
            dy_dr = torch.autograd.grad(
                y,
                data.pos,
                grad_outputs=torch.ones_like(y),
                retain_graph=self.training,
                create_graph=self.training,
            )[0]

            data.out[self.name][FORCE_KEY] = -dy_dr
            assert not torch.any(torch.isnan(dy_dr)), f"nan in {self.name}"
        return data
