mlcg-tools
==========

.. start-intro

This repository collects a set of tools to apply machine learning techniques to coarse grain atomic systems.

.. end-intro

Installation
------------
.. start-install

Some special installs that are one well handled by requirements.txt:

.. code:: bash

    conda install pyg=2.0.1 -c pyg -c conda-forge

    pip install git+https://github.com/PyTorchLightning/pytorch-lightning.git


.. end-install

Documentation
-------------

Dependencies
~~~~~~~~~~~~

.. code:: bash

    pip install sphinx sphinx_rtd_theme


Build instructions
~~~~~~~~~~~~~~~~~~

.. code:: bash

    cd docs
    sphinx-build -b html source build


