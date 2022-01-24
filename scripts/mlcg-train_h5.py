#! /usr/bin/env python

import sys
import os.path as osp
from time import ctime
import subprocess
import torch

SCRIPT_DIR = osp.abspath(osp.dirname(__file__))

sys.path.insert(0, osp.join(SCRIPT_DIR, "../"))

from mlcg.pl import PLModel, H5DataModule, LightningCLI


if __name__ == "__main__":
    # For avoiding 20 steps of painfully slow JIT recompilation
    # See https://github.com/pytorch/pytorch/issues/52286
    torch._C._jit_set_bailout_depth(3)
    git = {
        "log": subprocess.getoutput('git log --format="%H" -n 1 -z'),
        "status": subprocess.getoutput("git status -z"),
    }
    print("Start: {}".format(ctime()))

    cli = LightningCLI(
        PLModel,
        H5DataModule,
        save_config_overwrite=True,
        parser_kwargs={"error_handler": None},
    )

    print("Finish: {}".format(ctime()))
