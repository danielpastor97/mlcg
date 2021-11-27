Neighbor Lists
==============

`mlcg.neighborlist` contains several functions that can be used to create neighbor lists directly from data for features of arbitrary interaction order. There are also tools to interface with `ASE` representaions.

Neighbor List Utilities
-----------------------
These utilities are meant to format and validate neighbor list dictionaries.
.. autofunction:: mlcg.neighbor_list.neighbor_list.make_neighbor_list
.. autofunction:: mlcg.neighbor_list.neighbor_list.atom_data2neighbor_list
.. autofunction:: mlcg.neighbor_list.neighbor_list.validate_neighborlist


Torch Implementation
--------------------

.. autofunction:: mlcg.neighbor_list.torch_impl.torch_neighbor_list
.. autofunction:: mlcg.neighbor_list.torch_impl.torch_neighbor_list_no_pbc
.. autofunction:: mlcg.neighbor_list.torch_impl.compute_images
.. autofunction:: mlcg.neighbor_list.torch_impl.wrap_positions

ASE Implementation
------------------

.. autofunction:: mlcg.neighbor_list.ase_impl.ase_neighbor_list

