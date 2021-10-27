Training
========

`mlcg` provides some tools to train its models in the `scripts` folder and some example input files such as `examples/train_schnet.yaml`. The training is defined using the `pytorch-lightning <https://pytorch-lightning.readthedocs.io/en/latest/>`_ package and especially its `cli <https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html>`_ utilities.

Scripts
-------

Scripts that are using `LightningCLI` have many convinient builtin `functionalities <https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html#lightningcli>`_ such as a detailed helper

.. code:: bash

   python scripts/mlcg-train.py --help


Utils for using Pytorch Lightning
----------------------------------


.. autoclass:: mlcg.pl.PLModel

.. autoclass:: mlcg.pl.DataModule

.. autoclass:: mlcg.pl.LightningCLI

