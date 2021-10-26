import torch
import sys
import os
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import instantiate_class
from typing import Optional

from ..data import AtomicData
from ..nn import Loss, GradientsOut
from ._fix_hparams_saving import yaml


class PLModel(pl.LightningModule):
    """PL interface to train with models defined in :ref:`mlcg.nn`.

    Parameters
    ----------

        model:
            instance of a model class from :ref:`mlcg.nn`.
        loss:
            instance of :ref:`mlcg.nn.Loss`.
        optimizer:
            The optimizer to use to train the model in the form of `{"class_path":...[,"init_args":...]}`, where `"class_path"` is a class type, e.g. `"mlcg.nn.Cutoff"`, and `"init_args"` is a list of initialization arguments.
        lr_scheduler:
            The learning rate scheduler to use to train the model in the form of `{"class_path":...[,"init_args":...]}`, where `"class_path"` is a class type, e.g. `"mlcg.nn.Cutoff"`, and `"init_args"` is a list of initialization arguments.
        monitor:
            Metric to to monitor for schedulers like `ReduceLROnPlateau`.
        frequency:
            How many epochs/steps should pass between calls to
            `scheduler.step()`. 1 corresponds to updating the learning
            rate after every epoch/step.
        interval:
            The unit of the scheduler's step size, could also be 'step'.
            'epoch' updates the scheduler on epoch end whereas 'step'
            updates it after a optimizer update.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss: Loss,
        optimizer: dict = None,
        lr_scheduler: dict = None,
        monitor: Optional[str] = None,
        frequency: int = 1,
        interval: str = "epoch",
    ) -> None:
        """ """

        super(PLModel, self).__init__()

        self.save_hyperparameters()
        self.model = model
        self.loss = loss
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.monitor = monitor
        self.frequency = frequency
        self.interval = interval

        self.derivative = False
        for module in self.modules():
            if isinstance(module, GradientsOut):
                self.derivative = True

    def configure_optimizers(self) -> dict:
        optimizer = instantiate_class(self.model.parameters(), self.optimizer)
        scheduler = instantiate_class(optimizer, self.lr_scheduler)
        name = self.lr_scheduler["class_path"].split(".")[-1]
        if self.monitor is None:
            {"optimizer": optimizer, "lr_scheduler": scheduler, "name": name}
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": self.monitor,
                "frequency": self.frequency,
                "interval": self.interval,
                "name": name,
            }

    def training_step(self, data: AtomicData, batch_idx: int) -> torch.Tensor:
        loss = self.step(data, "training")
        return loss

    def validation_step(self, data: AtomicData, batch_idx) -> torch.Tensor:
        loss = self.step(data, "validation")
        return loss

    def test_step(self, data: AtomicData, batch_idx) -> torch.Tensor:
        loss = self.step(data, "test")
        return loss

    def step(self, data: AtomicData, stage: str) -> torch.Tensor:
        with torch.set_grad_enabled(stage == "train" or self.derivative):
            data = self.model(data)
        data.out.update(**data.out[self.model.name])
        loss = self.loss(data)

        # Add sync_dist=True to sync logging across all GPU workers
        self.log(
            f"{stage}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )

        return loss
