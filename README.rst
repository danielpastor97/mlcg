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

Documentation is available `here <https://clementigroup.github.io/mlcg-tools/>`_.

Dependencies
~~~~~~~~~~~~

.. code:: bash

    pip install sphinx sphinx_rtd_theme


How to build
~~~~~~~~~~~~

.. code:: bash

    cd docs
    sphinx-build -b html source build

How to update the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This udapte should be done after any update of the `main` branch so that the
documentation is synchronised with the main version of the repository.

.. code:: bash

    git checkout gh-pages
    git rebase main
    cd docs
    sphinx-build -b html source ./
    git commit -a
    git push