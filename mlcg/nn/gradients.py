import torch
from typing import Sequence, Dict, List
from ..data.atomic_data import AtomicData
from ..data._keys import *


class SumOut(torch.nn.Module):
    """Property pooling wrapper for models

    Parameters
    ----------
    models:
        Dictionary of predictors models keyed by their name attribute
    targets:
        List of prediction targets that will be pooled
    """

    def __init__(
        self,
        models: torch.nn.ModuleDict,
        targets: List[str] = [ENERGY_KEY, FORCE_KEY],
    ):
        super(SumOut, self).__init__()
        self.targets = targets
        self.models = models

    def forward(self, data: AtomicData) -> AtomicData:
        """Sums output properties from individual models into global
        property predictions

        Parameters
        ----------
        data:
            AtomicData instance whose 'out' field has been populated
            for each predictor in the model. For example:

        .. code-block::python

            AtomicData(
                out: {
                    SchNet: {
                        ENERGY_KEY: ...,
                        FORCE_KEY: ...,
                    },
                    bonds: {
                        ENERGY_KEY: ...,
                        FORCE_KEY: ...,
                    },

            ...
            )

        Returns
        -------
        data:
            AtomicData instance with updated 'out' field that now contains
            prediction target keys that map to tensors that have summed
            up the respective contributions from each predictor in the model.
            For example:

        .. code-block::python

            AtomicData(
                out: {
                    SchNet: {
                        ENERGY_KEY: ...,
                        FORCE_KEY: ...,
                    },
                    bonds: {
                        ENERGY_KEY: ...,
                        FORCE_KEY: ...,
                    },
                    ENERGY_KEY: ...,
                    FORCE_KEY: ...,
            ...
            )

        """
        for target in self.targets:
            data.out[target] = 0.00
        for name in self.models.keys():
            data = self.models[name](data)
            for target in self.targets:
                data.out[target] += data.out[name][target]
        return data


class GradientsOut(torch.nn.Module):
    """Gradient wrapper for models.

    Parameters
    ----------
    targets:
        The gradient targets to produce from a model output. These can be any
        of the gradient properties referenced in `mlcg.data._keys`.
        At the moment only forces are implemented.

    Example
    -------
        To preduct forces from an energy model, one woule supply a model that
        predicts a scalar atom property (an energy) and specify the FORCE_KEY
        in the targets.
    """

    _targets = {FORCE_KEY: ENERGY_KEY}

    def __init__(self, model: torch.nn.Module, targets: str = FORCE_KEY):
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

    def forward(self, data: AtomicData) -> AtomicData:
        """Forward pass through the gradient layer.

        Parameters
        ----------
        data:
            AtomicData instance

        Returns
        -------
        data:
            Updated AtomicData instance, where the "out" field has
            been populated with the base predictions of the model (eg,
            the energy as well as the target predictions produced through
            gradient operations.
        """

        if self.name not in data.out:
            data.out[self.name] = {}

        # if ENERGY_KEY not in data.out[self.name]:
        #     data.pos.requires_grad_(True)
        #     data = self.model(data)
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
        data.pos.requires_grad_(False)
        return data
