import os.path as osp
import torch
import pytorch_lightning.utilities.cli as plc
from torch_geometric.data.makedirs import makedirs


class LightningCLI(plc.LightningCLI):
    """Command line interface for training a model with pytorch lightning.

    It adds a few functionalities to `pytorch_lightning.utilities.cli.LightningCLI`.

    + register torch optimizers and lr_scheduler so that they can be specified
    in the configuration file. Note that only single (optimizer,lr_scheduler)
    can be specified like that and more complex patterns should be implemented
    in the pytorch_lightning model definition (child of `pytorch_lightning.
    LightningModule`). see `doc <https://pytorch-lightning.readthedocs.io/en/1.
    4.9/common/lightning_cli.html#optimizers-and-learning-rate-schedulers>`_
    for more details.

    + link manually some arguments related to the definition of the work directory. If `default_root_dir` argument of `pytorch_lightning.Trainer` is set and the `save_dir` / `log_dir` / `dirpath` argument of `loggers` / `data` / `callbacks` is set to `default_root_dir` then they will be set to the value of `default_root_dir` / `default_root_dir/data` / `default_root_dir/ckpt`.

    """

    def add_arguments_to_parser(self, parser):
        # TODO: remove when pl 1.5.0 is out as a stable release
        parser.add_optimizer_args(
            (
                torch.optim.AdamW,
                torch.optim.Adam,
                torch.optim.Adagrad,
                torch.optim.Adadelta,
                torch.optim.LBFGS,
            ),
            link_to="model.optimizer",
        )
        parser.add_lr_scheduler_args(
            (
                torch.optim.lr_scheduler.ExponentialLR,
                torch.optim.lr_scheduler.ReduceLROnPlateau,
                torch.optim.lr_scheduler.CosineAnnealingLR,
            ),
            link_to="model.lr_scheduler",
        )

    def parse_arguments(self) -> None:
        """Parses command line arguments and stores it in self.config"""
        self.config = self.parser.parse_args()

        trainer = self.config["trainer"]
        default_root_dir = trainer.get("default_root_dir")
        if default_root_dir is not None:
            for ii, logger in enumerate(trainer.get("logger", [])):
                save_dir = logger["init_args"].get("save_dir")
                if save_dir == "default_root_dir":
                    self.config["trainer"]["logger"][ii]["init_args"][
                        "save_dir"
                    ] = default_root_dir

            log_dir = self.config["data"].get("log_dir")
            if log_dir == "default_root_dir":
                path = osp.join(default_root_dir, "data")
                makedirs(path)
                self.config["data"]["log_dir"] = path

            for ii, callback in enumerate(trainer.get("callbacks", [])):
                dirpath = callback["init_args"].get("dirpath")
                if dirpath == "default_root_dir":
                    path = osp.join(default_root_dir, "ckpt")
                    makedirs(path)
                    self.config["trainer"]["callbacks"][ii]["init_args"][
                        "dirpath"
                    ] = path
