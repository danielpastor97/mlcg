import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import instantiate_class
from typing import Optional
from copy import deepcopy

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
        step_frequency:
            How many epochs/steps should pass between calls to
            `scheduler.step()`. 1 corresponds to updating the learning
            rate after every epoch/step.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss: Loss,
        optimizer: Optional[dict] = None,
        lr_scheduler: Optional[dict] = None,
        monitor: str = "validation_loss",
        step_frequency: int = 1,
    ) -> None:
        """ """

        super(PLModel, self).__init__()

        self.save_hyperparameters()
        self.model = model
        self.loss = loss
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.monitor = monitor
        self.step_frequency = step_frequency

        self.derivative = False
        for module in self.modules():
            if isinstance(module, GradientsOut):
                self.derivative = True

    def configure_optimizers(self) -> dict:
        optimizer = instantiate_class(
            self.model.parameters(), init=self.optimizer
        )
        scheduler = instantiate_class(optimizer, self.lr_scheduler)
        if self.monitor is None:
            {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": self.monitor,
                "frequency": self.step_frequency,
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

    def get_model(self) -> torch.nn.Module:
        return deepcopy(self.model)
