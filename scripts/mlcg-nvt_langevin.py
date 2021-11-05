import logging
import os
import time
import argparse
import numpy as np
import os.path as osp
import torch
import sys
from typing import Any, Dict, List, Optional
import inspect

from jsonargparse import (
    ActionConfigFile,
    ArgumentParser,
    class_from_function,
    set_config_read_mode,
)
from jsonargparse.actions import _ActionPrintConfig

SCRIPT_DIR = osp.abspath(osp.dirname(__file__))

sys.path.insert(0, osp.join(SCRIPT_DIR, "../"))

from mlcg.simulation import _Simulation, LangevinSimulation, O


class MisconfigurationException(Exception):
    """
    Exception used to inform users
    """


class Myparser(ArgumentParser):
    """Extension of jsonargparse's ArgumentParser"""

    def __init__(
        self,  # print_config: Optional[str] = '--print_config',
        *args: Any,
        parse_as_dict: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        For full details of accepted arguments see `ArgumentParser.__init__
        <https://jsonargparse.readthedocs.io/en/stable/#jsonargparse.core.ArgumentParser.__init__>`_.

        print_config: Add this as argument to print config, set None to disable.
        """

        super().__init__(*args, parse_as_dict=parse_as_dict, **kwargs)

        self.add_argument(
            "--config",
            type=ActionConfigFile,
            help="Path to a configuration file in json or yaml format.",
        )
        self.add_argument("--print_config", action=_ActionPrintConfig)
        # self._print_config = print_config

    def add_lightning_class_args(
        self,
        lightning_class: _Simulation,
        nested_key: str,
        subclass_mode: bool = False,
    ) -> List[str]:
        """
        Adds arguments from a lightning class to a nested key of the parser

        Args:
            lightning_class: A callable or any subclass of {Trainer, LightningModule, LightningDataModule, Callback}.
            nested_key: Name of the nested namespace to store arguments.
            subclass_mode: Whether allow any subclass of the given class.
        """
        nested_key
        if callable(lightning_class) and not inspect.isclass(lightning_class):
            lightning_class = class_from_function(lightning_class)

        if inspect.isclass(lightning_class) and issubclass(
            lightning_class, (_Simulation)
        ):
            if subclass_mode:
                return self.add_subclass_arguments(
                    lightning_class, nested_key, required=True
                )
            return self.add_class_arguments(
                lightning_class,
                nested_key,
                fail_untyped=False,
                instantiate=not issubclass(lightning_class, _Simulation),
            )
        raise MisconfigurationException(
            f"Cannot add arguments from: {lightning_class}. You should provide either a callable or a subclass of: "
            "Trainer, LightningModule, LightningDataModule, or Callback."
        )


def get_simulation_config(
    simulation_class: _Simulation,
    description: str = "Simulation command line tool",
    parser_kwargs: Dict[str, Any] = None,
):

    parser_kwargs = {} if parser_kwargs is None else parser_kwargs
    parser_kwargs.update({"description": description})
    parser = Myparser(**parser_kwargs)
    parser.add_lightning_class_args(
        simulation_class, "simulation", subclass_mode=False
    )

    parser.add_argument(
        "-mf",
        "--model-file",
        metavar="FN",
        type=str,
        help="path to the pytorch model file",
    )

    config = parser.parse_args()

    return config, parser


if __name__ == "__main__":

    config, parser = get_simulation_config(LangevinSimulation)
    model_fn = config.pop("model-file")
    print(model_fn)
    config_init = parser.instantiate_classes(config)

    simulation = config_init.get("simulation")
