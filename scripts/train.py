import sys
import os.path as osp
from time import ctime
import subprocess
import torch
from pytorch_lightning.utilities.cli import LightningCLI
from torch_geometric.data.makedirs import makedirs


SCRIPT_DIR = osp.abspath(osp.dirname(__file__))

sys.path.insert(0, osp.join(SCRIPT_DIR, "../"))


from mlcg.pl import PLModel, DataModule


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # parser.link_arguments("model.default_root_dir", "data.log_dir")

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
            path = osp.join(default_root_dir, 'data')
            makedirs(path)
            self.config["data"]["log_dir"] = path


        for ii, callback in enumerate(trainer.get("callbacks", [])):
            dirpath = callback['init_args'].get('dirpath')
            if dirpath == "default_root_dir":
                path = osp.join(default_root_dir, 'ckpt')
                makedirs(path)
                self.config["trainer"]["callbacks"][ii]["init_args"][
                        "dirpath"
                    ] = path

if __name__ == "__main__":

    git = {
        "log": subprocess.getoutput('git log --format="%H" -n 1 -z'),
        "status": subprocess.getoutput("git status -z"),
    }
    print("Start: {}".format(ctime()))

    cli = MyLightningCLI(
        PLModel,
        DataModule,
        save_config_overwrite=True,
    )

    print("Finish: {}".format(ctime()))
