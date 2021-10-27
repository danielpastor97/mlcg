import networkx as nx
import torch
import pytest
import numpy as np

from mlcg.nn.radial_basis import *

data = torch.randn(100, 3)
linear_data = torch.linspace(0, 10, 100)


@pytest.mark.parametrize("basis_type", [GaussianBasis, ExpNormalBasis])
def test_cutoff_error_raise(basis_type):
    """Test to make sure that RBFs enforce sensible cutoffs"""
    with pytest.raises(ValueError):
        basis_type(cutoff_lower=10, cutoff_upper=0)
