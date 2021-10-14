import torch
from typing import Sequence


class GradientsOut(torch.nn.Module):
    _targets = ["forces"]

    def __init__(self, energy_model, targets="forces"):
        super(GradientsOut, self).__init__()
        self.energy_model = energy_model
        self.name = self.energy_model.name
        self.targets = []
        if isinstance(targets, Sequence):
            self.targets = targets
        elif isinstance(targets, str):
            self.targets = [targets]
        assert any(
            [k in GradientsOut._targets for k in self.targets]
        ), f"targets={self.targets} should be any of {GradientsOut._targets}"

    def forward(self, data):
        if "forces" in self.targets:
            data.pos.requires_grad_(True)

        data = self.energy_model(data)

        if "forces" in self.targets:
            y = data.out["contribution"][self.name]["energy"]
            dy_dr = torch.autograd.grad(
                y,
                data.pos,
                grad_outputs=torch.ones_like(y),
                retain_graph=self.training,
                create_graph=self.training,
            )[0]

            data.out["contribution"][self.name]["forces"] = -dy_dr
            data.out["forces"] += -dy_dr
        return data
