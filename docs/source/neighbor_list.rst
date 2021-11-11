Neighbor Lists
==============

`mlcg.neighborlist` contains several functions that can be used to create neighbor lists directly from data for features of arbitrary interaction order. There are also tools to interface with `ASE` representaions.

Neighbor List Utilities
-----------------------

.. autofunction:: mlcg.neighbor_list.neighbor_list.make_neighbor_list
.. autofunction:: mlcg.neighbor_list.neighbor_list.atom_data2neighbor_list
.. autofunction:: mlcg.neighbor_list.neighbor_list.validate_neighborlist

ASE Implementation
------------------

.. autofunction:: mlcg.neighbor_list.ase_impl.ase_neighbor_list

Torch Implementation
--------------------

.. autofunction:: mlcg.neighbor_list.torch_impl.torch_neighbor_list
.. autofunction:: mlcg.neighbor_list.torch_impl.torch_neighbor_list_no_pbc
.. autofunction:: mlcg.neighbor_list.torch_impl.compute_images
.. autofunction:: mlcg.neighbor_list.torch_impl.wrap_positions
