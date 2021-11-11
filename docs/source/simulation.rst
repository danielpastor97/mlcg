Simulations
===========

`mlcg` provides some tools to run simulations with its models in the `scripts` folder and some example input files such as `examples/langevin.yaml`.


Utils for using Pytorch Lightning
----------------------------------

These classes define simulation integration schemes. After setting simulation options, attaching a model, and attaching initial configurations, they can additionally be used to run CG simulations on CPU or GPU.

.. autoclass:: mlcg.simulation.LangevinSimulation
.. autoclass:: mlcg.simulation.OverdampedSimulation



