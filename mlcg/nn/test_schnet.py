import networkx as nx
import torch
import pytest
import numpy as np

from mlcg.nn.cutoff import *

@pytest.mark.parametrize("cutoff_type", [IdentityCutoff, CosineCutoff])
def test_cutoff_error_raise(cutoff_type):
    with pytest.raises(ValueError):
        cutoff_type(cutoff_lower=10, cutoff_upper=0)
