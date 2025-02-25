import torch
from warnings import warn
from .prior import _Prior, Dihedral, Harmonic, Repulsion


def sparsify_prior_module(module: _Prior) -> torch.nn.Module:
    r"""
    Converts parameter tensors inplace to sparse tensors in prior objects
    """
    if isinstance(module, Dihedral):
        module.v_0 = module.v_0.to_sparse()
        module.k1s = module.k1s.to_sparse()
        module.k2s = module.k2s.to_sparse()
    elif isinstance(module, Repulsion):
        module.sigma = module.sigma.to_sparse()
    elif issubclass(type(module), Harmonic):
        module.x_0 = module.x_0.to_sparse()
        module.k = module.k.to_sparse()
    else:
        warn(
            f"Input is not a prior subclass. parameter will be retunerd as is"
        )
    return module
