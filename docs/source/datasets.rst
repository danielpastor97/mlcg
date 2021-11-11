Datasets
========

`mlcg.datasets` contains a template `InMemoryDataset` for CLN025, a 10 amino acid long mini protein that shows prototypical folding and unfolding behavior. The `ChignolinDataset` class illustrates how a general dataset can be downloaded, unpacked/organized, transformed/processed, and collated for training models. (Here)[https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html], users can find more information on implementing custom datasets.

Chignolin Dataset
-----------------

.. autoclass:: mlcg.datasets.chignolin.ChignolinDataset

Coarse Graining Utilities
-------------------------

.. autofunction:: mlcg.datasets.utils.remove_baseline_forces

