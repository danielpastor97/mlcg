mlcg
==========

.. start-intro

This repository collects a set of tools to apply machine learning techniques to coarse grain atomic systems.


Installation
------------
.. start-install
The dependencies are defined in `requirements.txt` but some packages are not well handled by pip. So start by installing `pytorch <https://pytorch.org/>`_ and `pytorch-geometric <https://pytorch-geometric.readthedocs.io/en/latest/>`_ with conda, e.g.::

    conda install pytorch cudatoolkit=11.3 -c pytorch
    conda install pyg -c pyg -c conda-forge

Support for the MACE model can be enabled with::

    pip install git+https://github.com/felixmusil/mace.git@develop

Support for the TorchMD-Net models can be enabled with::

    pip install git+https://github.com/felixmusil/torchmd-net.git

.. end-install

CLI
---

The models defined in this library can be convinietly trained using the pytorch-lightning
`cli <https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html>`_ utilities.

.. end-intro

.. start-doc

Documentation
-------------

Documentation is available `here <https://clementigroup.github.io/mlcg-tools/>`_ and here are some references on how to work with it.

Dependencies
~~~~~~~~~~~~

.. code:: bash

    pip install sphinx sphinx_rtd_theme sphinx-autodoc-typehints


How to build
~~~~~~~~~~~~

.. code:: bash

    cd docs
    sphinx-build -b html source build

How to update the online documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This udapte should be done after any update of the `main` branch so that the
documentation is synchronised with the main version of the repository.

.. code:: bash

    git checkout gh-pages
    git rebase main
    cd docs
    sphinx-build -b html source ./
    git commit -a
    git push

.. end-doc

Test Coverage
-------------

The test coverage of this library is monitored with `coverage` for each pull requests using `github` actions.
To produce a report locally run:

.. code:: bash
    coverage run -m pytest
    coverage report

test
