from typing import Any, List, Dict
import torch
from jsonargparse import (
    ActionConfigFile,
    ArgumentParser,
    class_from_function,
)

from .base import _Simulation


def parse_simulation_config(
    simulation_class,
    description: str = "Simulation command line tool",
    parser_kwargs: Dict[str, Any] = None,
):

    parser_kwargs = {} if parser_kwargs is None else parser_kwargs
    parser_kwargs.update({"description": description})
    parser = SimulationParser(**parser_kwargs)
    parser.add_simulation_args(
        simulation_class, "simulation", subclass_mode=False
    )
    parser.add_argument(
        "-mf",
        "--model_file",
        metavar="FN",
        type=str,
        help="path to the pytorch model file (including the priors) in pytorch format",
    )

    parser.add_argument(
        "-sf",
        "--structure_file",
        metavar="FN",
        type=str,
        help="path to the starting configurations in pytorch format",
    )

    config = parser.parse_args()

    model_fn = config.pop("model_file")
    model = torch.load(model_fn)

    structures_fn = config.pop("structure_file")
    initial_data_list = torch.load(structures_fn)
    config_init = parser.instantiate_classes(config)
    simulation = config_init.get("simulation")

    return model, initial_data_list, simulation


class MisconfigurationException(Exception):
    """
    Exception used to inform users
    """


class SimulationParser(ArgumentParser):
    """Extension of jsonargparse's ArgumentParser to parse simulation arguments."""

    def __init__(
        self,
        *args: Any,
        parse_as_dict: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        For full details of accepted arguments see `ArgumentParser.__init__
        <https://jsonargparse.readthedocs.io/en/stable/#jsonargparse.core.ArgumentParser.__init__>`_.

        """

        super().__init__(*args, parse_as_dict=parse_as_dict, **kwargs)

        self.add_argument(
            "--config",
            action=ActionConfigFile,
            help="Path to a configuration file in json or yaml format.",
        )

    def add_simulation_args(
        self,
        simulation_class: _Simulation,
        nested_key: str,
        subclass_mode: bool = False,
        required: bool = True,
    ) -> List[str]:
        """
        Adds arguments from a lightning class to a nested key of the parser

        Parameters
        ----------

        simulation_class:
            A callable or any subclass of {_Simulation}.
        nested_key:
            Name of the nested namespace to store arguments.
        subclass_mode:
            Whether allow any subclass of the given class.
        """
        if callable(simulation_class) and not isinstance(
            simulation_class, type
        ):
            simulation_class = class_from_function(simulation_class)

        if isinstance(simulation_class, type) and issubclass(
            simulation_class, (_Simulation)
        ):
            if subclass_mode:
                return self.add_subclass_arguments(
                    simulation_class,
                    nested_key,
                    fail_untyped=False,
                    required=required,
                )
            return self.add_class_arguments(
                simulation_class,
                nested_key,
                fail_untyped=False,
                instantiate=True,
                sub_configs=True,
            )
        raise MisconfigurationException(
            f"Cannot add arguments from: {simulation_class}. You should provide either a callable or a subclass of: "
            "Trainer, LightningModule, LightningDataModule, or Callback."
        )
