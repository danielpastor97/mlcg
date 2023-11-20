Simulations
===========

`mlcg` provides some tools to run simulations with its models in the `scripts` folder and some example input files such as `examples/langevin.yaml`.


NVT ensemble
------------

These classes define simulation integration schemes. After setting simulation options, attaching a model, and attaching initial configurations, they can additionally be used to run CG simulations on CPU or GPU.

.. autoclass:: mlcg.simulation.LangevinSimulation
.. autoclass:: mlcg.simulation.OverdampedSimulation
.. autoclass:: mlcg.simulation.PTSimulation


Utilities
---------

.. automodule:: mlcg.simulation.specialize_prior
