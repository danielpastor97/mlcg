import os.path as osp
import sys
import time
import argparse

import torch

SCRIPT_DIR = osp.abspath(osp.dirname(__file__))

sys.path.insert(0, osp.join(SCRIPT_DIR, "../"))

from mlcg.pl import PLModel


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="frobble")

    parser.add_argument(
        "-cf",
        "--checkpoint-file",
        metavar="FN",
        type=str,
        help="path to the pytorch lightning checkpoint file",
    )

    parser.add_argument(
        "-hf",
        "--hparams-file",
        metavar="FN",
        type=str,
        help="path to the pytorch lightning hparams file",
        default=None,
    )

    parser.add_argument(
        "-hf",
        "--hparams-file",
        metavar="FN",
        type=str,
        help="path to the pytorch lightning hparams file",
        default=None,
    )

    config = parser.parse_args()
    checkpoint_path = config.checkpoint_file
    hparams_file = config.hparams_file

    plmodel = PLModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path, hparams_file=hparams_file
    )
    # TODO load list of AtomicData
