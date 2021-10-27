import networkx as nx
import torch
import pytest
import numpy as np

from mlcg.nn import *

standard_basis = GaussianBasis()
standard_cutoff = IdentityCutoff()


@pytest.mark.parametrize(
    "basis, cutoff, expected_warning",
    [
        (GaussianBasis(0, 5), CosineCutoff(0, 5), None),
        (GaussianBasis(1, 5), CosineCutoff(0, 5), UserWarning),
    ],
)
def test_cutoff_warning(basis, cutoff, expected_warning):
    with pytest.warns(expected_warning):
        StandardSchNet(basis, cutoff, [128, 128])


def test_minimum_interaction_block():
    with pytest.raises(ValueError):
        StandardSchNet(
            standard_basis, standard_cutoff, [128, 128], num_interactions=-1
        )
