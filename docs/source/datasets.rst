Datasets
========

`mlcg.datasets` contains a template `InMemoryDataset` for CLN025, a 10 amino acid long mini protein that shows prototypical folding and unfolding behavior. The `ChignolinDataset` class illustrates how a general dataset can be downloaded, unpacked/organized, transformed/processed, and collated for training models. `Here <https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html>`_, users can find more information on implementing custom datasets.

Chignolin Dataset
-----------------

Dataset of 3744 individual all-atom trajectories simulated using [ACEMD]_ using the adaptive sampling strategy described in [AdaptiveStrategy]_ . All trajectories were simulated at 350K with [CHARM22Star]_ in a cubic box of :math:`\AA^3` with 1881 TIP3P water molecules and two Na:sup:`+` ions using a Langevin integrator with an integration timestep of 4 fs, a damping constant of 0.1:sup:`-1` ps, heavy-hydrogen constraints, and a PME cutoff of 9 :math:`\AA` and a PME mesh grid of 1 :math:`\AA`. The total aggregate simulation time is 187.2 us. 

.. autoclass:: mlcg.datasets.chignolin.ChignolinDataset

Coarse Graining Utilities
-------------------------

.. autofunction:: mlcg.datasets.utils.remove_baseline_forces

