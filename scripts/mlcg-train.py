#! /usr/bin/env python

import sys
import os.path as osp
from time import ctime
import subprocess


SCRIPT_DIR = osp.abspath(osp.dirname(__file__))

sys.path.insert(0, osp.join(SCRIPT_DIR, "../"))

from mlcg.pl import PLModel, DataModule, LightningCLI


if __name__ == "__main__":

    git = {
        "log": subprocess.getoutput('git log --format="%H" -n 1 -z'),
        "status": subprocess.getoutput("git status -z"),
    }
    print("Start: {}".format(ctime()))

    cli = LightningCLI(
        PLModel,
        DataModule,
        save_config_overwrite=True,
    )

    print("Finish: {}".format(ctime()))
