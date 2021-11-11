Models
======

`mlcg.nn` currently implements the SchNet graph neural network, as well as several utility classes for computing distance expansions and cutoffs. The `nn` subpackage also contains several useful classes for extracting other properties from energy predictions or aggregating the predictions from several different model types.

SchNet Utilities
----------------

These classes are used to define a SchNet graph neural network. For "typical" SchNet models, users
may find the class `StandardSchNet` to be helpful in getting started quickly.

.. autoclass:: mlcg.nn.schnet.SchNet
.. autoclass:: mlcg.nn.schnet.InteractionBlock
.. autoclass:: mlcg.nn.CFConv
.. autoclass:: mlcg.nn.StandardSchNet

Radial Basis Functions
----------------------

These classes are used to expanded pairwsie distances into a basis of more general input features for filter generating networks in `SchNet`.

.. autoclass:: mlcg.nn.radial_basis.GaussianBasis
.. autoclass:: mlcg.nn.radial_basis.ExpNormalBasis

Cutoff Functions
----------------

These classes are used to apply cutoffs and envelops to `CFConv` inputs.

.. autoclass:: mlcg.nn.cutoff.IdentityCutoff
.. autoclass:: mlcg.nn.cutoff.CosineCutoff

Model Building Utilities
------------------------

These classes are used to build more complicated models.

.. autoclass:: mlcg.nn.gradients.GradientsOut
.. autoclass:: mlcg.nn.gradients.SumOut

Loss Functions
--------------

These classes define loss functions for model optimization, as well as generalized losses that combine several losses of different types.

.. autoclass:: mlcg.nn.losses.Loss
.. autoclass:: mlcg.nn.losses.ForceRMSE
