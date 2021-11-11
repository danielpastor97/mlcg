Priors
======

`mlcg.nn` implements prior models that are important for imposing geometrical constraints on coarse grain systems for both training and simulation. These prior models can be supplied with a user defined set of interaction parameters, or they can alternatively be parametrized directly from data for multiple interacting atom types. 

Prior Utilities
----------------

These classes define several common molecular interactions. Because each prior subclasses `torch.nn.Module`, they can be treated as normal property predictors.

.. autoclass:: mlcg.nn.prior.HarmonicBonds
.. autoclass:: mlcg.nn.prior.HarmonicAngles
.. autoclass:: mlcg.nn.prior.Repulsion
