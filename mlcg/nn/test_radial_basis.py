import torch
import pytest
from mlcg.nn.radial_basis import GaussianBasis, ExpNormalBasis
from mlcg.nn.cutoff import IdentityCutoff, CosineCutoff


@pytest.mark.parametrize("basis_type", [GaussianBasis, ExpNormalBasis])
def test_cutoff_error_raise(basis_type):
    """Test to make sure that RBFs enforce sensible cutoffs"""
    with pytest.raises(ValueError):
        basis_type(cutoff=IdentityCutoff(cutoff_lower=10, cutoff_upper=0))


@pytest.mark.parametrize(
    "basis_type, default_cutoff",
    [(GaussianBasis, IdentityCutoff), (ExpNormalBasis, CosineCutoff)],
)
def test_cutoff_defaults(basis_type, default_cutoff):
    cutoff_upper = 10
    basis = basis_type(cutoff=cutoff_upper)
    assert isinstance(basis.cutoff, default_cutoff)
    assert basis.cutoff.cutoff_lower == 0
    assert basis.cutoff.cutoff_upper == cutoff_upper
