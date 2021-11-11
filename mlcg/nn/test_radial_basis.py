import torch
import pytest
from mlcg.nn.radial_basis import GaussianBasis, ExpNormalBasis


@pytest.mark.parametrize("basis_type", [GaussianBasis, ExpNormalBasis])
def test_cutoff_error_raise(basis_type):
    """Test to make sure that RBFs enforce sensible cutoffs"""
    with pytest.raises(ValueError):
        basis_type(cutoff_lower=10, cutoff_upper=0)
