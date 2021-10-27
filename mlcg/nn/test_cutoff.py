import networkx as nx
import torch
import pytest
import numpy as np

from mlcg.nn.cutoff import *

data = torch.randn(100,3)
uniform_data = torch.ones(100,3)

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

@pytest.mark.parametrize("cutoff_lower, cutoff_upper, expected_lower, expected_upper", [(0,5,1,0),(5,10,0,0)])
def test_cosine_cutoff(cutoff_lower, cutoff_upper, expected_lower_value, expected_upper_value):
    """Test to make sure that CosineCutoff produces the correct cutoff
    envelope for different lower and upper cutoffs - the cutoff endpoints change
    values for non-zero lower cutoffs
    cutoff = CosineCutoff(cutoff_lower=cutoff_lower, cutoff_upper=cutoff_upper)
    cutoff_data = cutoff(data).numpy()
    expected_data = expected_lower_value(
    cutoff_data[0]
    """
    pass
