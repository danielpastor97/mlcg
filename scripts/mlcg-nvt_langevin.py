#! /usr/bin/env python

from time import ctime
import os.path as osp
import torch
import sys
from typing import Any, Dict

SCRIPT_DIR = osp.abspath(osp.dirname(__file__))

sys.path.insert(0, osp.join(SCRIPT_DIR, "../"))

from mlcg.simulation import (
    parse_simulation_config,
    LangevinSimulation,
)


if __name__ == "__main__":
    print(f"Starting simulation at {ctime()} with {LangevinSimulation}")
    (
        model,
        initial_data_list,
        betas,
        simulation,
        profile,
    ) = parse_simulation_config(LangevinSimulation)

    simulation.attach_model_and_configurations(
        model, initial_data_list, beta=betas
    )
    simulation.simulate()
    print(f"Ending simulation at {ctime()}")
