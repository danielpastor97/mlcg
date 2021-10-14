import torch
from torch_geometric.data import Data
from typing import Optional, Dict, Any

from ._keys import (
    validate_keys,
    POSITIONS_KEY,
    N_ATOMS_KEY,
    CELL_KEY,
    PBC_KEY,
    ATOM_TYPE_KEY,
    NEIGHBOR_LIST_KEY,
    TAG_KEY,
    ENERGY_KEY,
    FORCE_KEY,
)


class AtomicData(Data):
    """A data object holding atomic structures.

    Attributes
    ----------

    kwargs:
        Allowed fields are defined in :ref:`ALLOWED_KEYS`


    """

    def __init__(
        self,
        **kwargs,
    ):
        validate_keys(kwargs.keys())

        super(AtomicData, self).__init__(**kwargs)

        self.out = {}

        # check the sanity of the inputs
        if "n_atoms" in self and "pos" in self:
            assert (
                torch.sum(self.n_atoms) == self.pos.shape[0]
            ), f"number of atoms {torch.sum(self.n_atoms)} and number of positions {self.pos.shape[0]}"

            assert self.pos.shape[1] == 3

        if CELL_KEY in self and self.cell is not None:
            assert self.cell.dim() == 3
            assert self.cell.shape[1:] == torch.Size((3, 3))
            assert self.cell.dtype == self.pos.dtype
        if FORCE_KEY in self and self[FORCE_KEY] is not None:
            assert self[FORCE_KEY].shape == self[POSITIONS_KEY].shape
            assert self[FORCE_KEY].dtype == self[POSITIONS_KEY].dtype
        if ENERGY_KEY in self and self[ENERGY_KEY] is not None:
            assert self[ENERGY_KEY].shape == self[N_ATOMS_KEY].shape
            assert self[ENERGY_KEY].dtype == self[POSITIONS_KEY].dtype
        if PBC_KEY in self and self.pbc is not None:
            assert self.pbc.dim() == 2, f"dim {self.pbc.dim()}"
            assert self.pbc.shape[1:] == torch.Size(
                [3]
            ), f"shape {self.pbc.shape[1:]}"
            assert self.pbc.dtype == torch.bool

    @staticmethod
    def from_ase(
        frame,
        energy_tag: str = ENERGY_KEY,
        force_tag: str = FORCE_KEY,
    ):
        z = torch.from_numpy(frame.get_atomic_numbers())
        pos = torch.from_numpy(frame.get_positions())
        pbc = torch.from_numpy(frame.get_pbc())
        cell = torch.tensor(frame.get_cell().tolist(), dtype=torch.float64)

        tag = frame.info.get("tag")

        energy = frame.info.get(energy_tag)
        forces = frame.arrays.get(force_tag)

        return AtomicData.from_points(
            pos=pos,
            atom_types=z,
            pbc=pbc,
            cell=cell,
            tag=tag,
            energy=energy,
            forces=forces,
        )

    @staticmethod
    def from_points(
        pos: torch.Tensor,
        atom_types: torch.Tensor,
        pbc: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
        tag: Optional[str] = None,
        energy: Optional[torch.Tensor] = None,
        forces: Optional[torch.Tensor] = None,
        neighborlist: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs,
    ):
        data = {}
        data.update(**kwargs)
        data[ATOM_TYPE_KEY] = torch.as_tensor(atom_types)
        data[POSITIONS_KEY] = torch.as_tensor(pos)
        data[N_ATOMS_KEY] = torch.tensor(
            [data[ATOM_TYPE_KEY].shape[0]], dtype=torch.long
        )
        if energy is not None:
            data[ENERGY_KEY] = torch.as_tensor(
                energy, dtype=data[POSITIONS_KEY].dtype
            )
        if forces is not None:
            data[FORCE_KEY] = torch.as_tensor(
                forces, dtype=data[POSITIONS_KEY].dtype
            )
        data[TAG_KEY] = tag
        if neighborlist is None:
            data[NEIGHBOR_LIST_KEY] = {}
        else:
            data[NEIGHBOR_LIST_KEY] = neighborlist

        if pbc is not None:
            data[PBC_KEY] = torch.as_tensor(pbc).view(-1, 3)
        if cell is not None:
            data[CELL_KEY] = torch.as_tensor(cell).view(-1, 3, 3)

        return AtomicData(**data)
