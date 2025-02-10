import pytest
import torch
from .seq_neigh import get_seq_neigh
from ..data.atomic_data import AtomicData

def simple_struct_1():
    n = 10
    pos = torch.randn((n,3))
    atom_types = torch.randint(20,(n,))
    masses = torch.ones(10)
    batch = torch.tensor([0,0,0,1,1,1,1,2,2,2])
    data = AtomicData.from_points(
        pos=pos,
        atom_types=atom_types,
        masses=masses,
    )
    data.batch = batch
    expected_neighs = torch.tensor([[0, 1, 1, 2, 3, 4, 4, 5, 5, 6, 7, 8, 8, 9],
        [1, 2, 0, 1, 4, 5, 3, 6, 4, 5, 8, 9, 7, 8]])
    return data, expected_neighs

def test_seq_neigh():
    atom_data, expected_neighs = simple_struct_1()
    answer = get_seq_neigh(atom_data)
    # convert them to sets of tuples to make comparison
    expected_neighs_set = set([tuple(row.numpy()) for row in expected_neighs.T])
    answer_set = set([tuple(row.numpy()) for row in answer.T])
    assert (answer_set == expected_neighs_set)
