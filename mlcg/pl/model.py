import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import instantiate_class
from pytorch_lightning.utilities.finite_checks import detect_nan_parameters
from typing import Optional, Tuple
from copy import deepcopy

from ..data import AtomicData
from ..data._keys import N_ATOMS_KEY
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
        monitor: Optional[str] = None,
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
        optim_config = {"optimizer": optimizer}

        if self.lr_scheduler is not None:
            scheduler = instantiate_class(optimizer, self.lr_scheduler)
            optim_config.update(
                lr_scheduler=scheduler,
            )
            if self.monitor is not None:
                optim_config.update(
                    monitor=self.monitor,
                    frequency=self.step_frequency,
                )

        return optim_config

    def on_epoch_start(self):
        # this can avoid growing VRAM usage after 1 epoch
        # so that we can maximize the batch size quickly on a GPU
        # by the success of first several training steps
        torch.cuda.empty_cache()

    def on_train_epoch_end(self):
        # instead of checking it after every training step, which is costly
        detect_nan_parameters(self.model)

    def validation_epoch_end(self, validation_step_outputs):
        # we calculate the epochal validation loss that is compatible with
        # the original loss calculation for combined metasets (which is still used for training)
        # i.e., a weighted mean with batch size for each metaset as weight
        with torch.no_grad():
            out = torch.tensor(
                validation_step_outputs
            )  # shape (N_metasets, N_batches, 2)
            # if metasets are not used, or if just one metaset is used,
            # adjust the shape accordingly:
            if len(out.shape) != 3:
                out = out.view(1, *out.shape)
            out = (
                out[:, :, 0]
                * out[:, :, 1]
                / out[:, :, 1].sum(axis=0, keepdim=True)
            )
            out = out.sum(axis=0)  # shape (N_batches,)
            out = out.mean()
        # in ddp this will be run separately on different processes
        # therefore we need to synchronize (sync_dist=True)
        self.log(
            f"validation_loss",
            out,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )

    def training_step(self, data: AtomicData, batch_idx: int) -> torch.Tensor:
        loss, _ = self.step(data, "training")
        return loss

    def validation_step(
        self, data: AtomicData, batch_idx, dataloader_idx=0
    ) -> Tuple[torch.Tensor, int]:
        loss, batch_size = self.step(data, "validation")
        return loss, batch_size

    def test_step(
        self, data: AtomicData, batch_idx, dataloader_idx=0
    ) -> Tuple[torch.Tensor, int]:
        loss, batch_size = self.step(data, "test")
        return loss, batch_size

    def step(self, data: AtomicData, stage: str) -> Tuple[torch.Tensor, int]:
        with torch.set_grad_enabled(stage == "train" or self.derivative):
            data = self.model(data)
        data.out.update(**data.out[self.model.name])
        loss = self.loss(data)
        batch_size = data[N_ATOMS_KEY].shape[0]
        # Add sync_dist=True to sync logging across all GPU workers
        self.log(
            f"{stage}_loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        return loss, batch_size

    def get_model(self) -> torch.nn.Module:
        return deepcopy(self.model)
