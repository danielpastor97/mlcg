mlcg-tools
==========

.. image:: https://codecov.io/gh/ClementiGroup/mlcg-tools/branch/main/graph/badge.svg?token=TXZNC7X73E
     :target: https://codecov.io/gh/ClementiGroup/mlcg-tools
    
.. start-intro

This repository collects a set of tools to apply machine learning techniques to coarse grain atomic systems.

.. end-intro

Installation
------------
.. start-install

Some special installs that are one well handled by requirements.txt:

.. code:: bash

    conda install pyg=2.0.1 -c pyg -c conda-forge

.. end-install

CLI
---

The models defined in this library can be convinietly trained using the pytorch-lightning
`cli <https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html>`_ utilities.


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
