import torch
from torch_geometric.data import Data
from typing import Optional, Dict, NamedTuple

from ._keys import (
    ALLOWED_KEYS,
    POSITIONS_KEY,
    N_ATOMS_KEY,
    CELL_KEY,
    PBC_KEY,
    ATOM_TYPE_KEY,
    NEIGHBOR_LIST_KEY,
)

from ..neighbor_list import torch_neighbor_list



class NeighborList(NamedTuple):
    """data structure holding the information about connectivity within atomic
    structures.
    """
    #: quick identifier for compatibility checking
    tag: str
    #: an int providing the order of the neighborlist, e.g. order == 2 means that
    #: central atoms `i` have 1 neighbor `j` so distances can be computed,
    #: order == 3 means that central atoms `i` have 2 neighbors `j` and `k` so
    #: angles can be computed
    order: int
    #: The [2, n_edge] index tensor giving center -> neighbor relations. 1st column
    #: refers to the central atom index and the 2nd column to the neighbor atom
    #: index in the list of atoms (so it has to be shifted by a cell_shift to get
    #: the actual position of the neighboring atoms)
    mapping: torch.Tensor
    #: A [n_edge, 3] tensor giving the periodic cell shift
    cell_shifts: Optional[torch.Tensor] = None
    #: cutoff radius used to compute the connectivity
    rcut: Optional[float] = None
    #: wether the mapping includes self refferring mappings, e.g. mappings where
    #: `i` == `j`.
    self_interaction: Optional[bool] = None




class AtomicData(Data):
    """A data object holding atomic structures.

    Attributes
    ----------

    kwargs:
        Allowed fields are defined in :ref:`ALLOWED_KEYS`


    """

    def __init__(
        self,
        self_interaction: bool = False,
        **kwargs,
    ):
        super(AtomicData, self).__init__(**kwargs)

        if CELL_KEY in self and self.cell is not None:
            assert (self.cell.shape == (3, 3)) or (
                self.cell.dim() == 3 and self.cell.shape[1:] == (3, 3)
            )
            assert self.cell.dtype == self.pos.dtype

        if PBC_KEY in self and self.pbc is not None:
            assert (self.pbc.shape == (3)) or (
                self.pbc.dim() == 3 and self.pbc.shape[1:] == (3)
            )
            assert self.pbc.dtype == torch.bool

    @staticmethod
    def from_ase(
        cls,
        frame,
    ):
        z = torch.from_numpy(frame.get_atomic_numbers())
        pos = torch.from_numpy(frame.get_positions())
        pbc = torch.from_numpy(frame.get_pbc())
        cell = torch.tensor(frame.get_cell().tolist(), dtype=torch.float64)

        tag = frame.info.get('tag')

        return AtomicData.from_points(
            pos=pos,
            atomic_type=z,
            pbc=pbc,
            cell=cell,
            tag=tag,
        )

    @staticmethod
    def from_points(
        cls,
        pos: torch.Tensor,
        atomic_type: torch.Tensor,
        pbc: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
        tag: Optional[str] = None,
        neighborlist: Optional[Dict[str, NeighborList]] = None,
    ):
        data = {}
        data[ATOM_TYPE_KEY] = torch.as_tensor(atomic_type)
        data[POSITIONS_KEY] = torch.as_tensor(pos)
        data[N_ATOMS_KEY] = data[ATOM_TYPE_KEY].shape[0]

        data['tag'] = tag
        if neighborlist is None:
            data[NEIGHBOR_LIST_KEY] = {}
        else:
            data[NEIGHBOR_LIST_KEY] = neighborlist

        if pbc is not None:
            data[PBC_KEY] = torch.as_tensor(pbc)
        if cell is not None:
            data[CELL_KEY] = torch.as_tensor(cell)

        return cls(**data)

    