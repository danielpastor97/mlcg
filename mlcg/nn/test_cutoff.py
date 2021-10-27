import networkx as nx
import torch
import pytest
import numpy as np

from mlcg.nn.cutoff import *

data = torch.randn(100,3)

@pytest.mark.parametrize("cutoff_type", [IdentityCutoff, CosineCutoff])
def test_cutoff_error_raise(cutoff_type):
    """Test to make sure that cutoffs enforce sensible cutoffs"""
    with pytest.raises(ValueError):
        cutoff_type(cutoff_lower=10, cutoff_upper=0)

def test_identity_cutoff():
    """Test to make sure that IdentityCutoff performs an identity transform"""
    cutoff = IdentityCutoff()
    data_out = cutoff(data)
    np.testing.assert_array_equal(data.numpy(), data_out.numpy())

