from typing import Optional
from numpy import zeros
import torch
from .torch_impl import torch_neighbor_list


def atomic_data2neighbor_list(
    data,
    rcut: float,
    self_interaction: bool = False,
) -> dict:
    """Build a neighborlist from a :ref:`mlcg.data.atomic_data.AtomicData` by
    searching for neighboring atom within a maximum radius `rcut`.

    Parameters
    ----------
    data: :ref:`mlcg.data.atomic_data.AtomicData`
        define an atomic structure
    rcut:
        cutoff radius used to compute the connectivity
    self_interaction:
        whether the mapping includes self referring mappings, e.g. mappings where `i` == `j`.
    """
    rcut = float(rcut)
    idx_i, idx_j, cell_shifts, _ = torch_neighbor_list(
        data, rcut, self_interaction=self_interaction
    )

    mapping = torch.cat([idx_i.unsqueeze(0), idx_j.unsqueeze(0)], dim=0)
    order = mapping.shape[0]
    return make_neighbor_list(
        tag=f"nonbounded rc:{rcut} order:{order}",
        order=order,
        index_mapping=mapping,
        cell_shifts=cell_shifts,
        rcut=rcut,
        self_interaction=self_interaction,
    )


def make_neighbor_list(
    tag: str,
    order: int,
    index_mapping: torch.Tensor,
    cell_shifts: Optional[torch.Tensor] = None,
    rcut: Optional[float] = None,
    self_interaction: Optional[bool] = None,
):
    """Build a neighbor_list dictionary.

    Parameters
    ----------
    tag:
        quick identifier for compatibility checking
    order:
        an int providing the order of the neighborlist, e.g. order == 2 means that
        central atoms `i` have 1 neighbor `j` so distances can be computed,
        order == 3 means that central atoms `i` have 2 neighbors `j` and `k` so
        angles can be computed
    index_mapping:
        The [order, n_edge] index tensor giving center -> neighbor relations. 1st column
        refers to the central atom index and the 2nd column to the neighbor atom
        index in the list of atoms (so it has to be shifted by a cell_shift to get
        the actual position of the neighboring atoms)
    cell_shifts:
        A [n_edge, 3] tensor giving the periodic cell shift
    rcut:
        cutoff radius used to compute the connectivity
    self_interaction:
        whether the mapping includes self referring mappings, e.g. mappings where `i` == `j`.
    """

    return dict(
        tag=tag,
        order=order,
        index_mapping=index_mapping,
        cell_shifts=cell_shifts,
        rcut=rcut,
        self_interaction=self_interaction,
    )


class _EmptyField:
    pass


def validate_neighborlist(inp: dict) -> bool:
    """Tool to validate that the neighborlist dictionary has the required fields"""
    validator = {
        "tag": [str],
        "order": [int],
        "index_mapping": [torch.Tensor],
        "cell_shifts": [torch.Tensor],
        "rcut": [float],
        "self_interaction": [bool],
    }
    for k, ts in validator.items():
        v = inp.get(k, _EmptyField())
        assert all(
            [isinstance(v, t) for t in ts]
        ), f"entry {k} is {type(v)} but should be in {ts}"
    return True
