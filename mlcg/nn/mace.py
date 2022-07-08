import torch
from typing import Final
from e3nn import o3
import numpy as np

try:
    from mace.modules.models import MACE
    from mace import modules
    from torch_geometric.data import Batch
    from mace.tools import to_one_hot

except ImportError:
    print(
        """Please install or set mace to your path before using this interface.
    To install you can either run
    'pip install git+https://github.com/ACEsuit/mace.git'
    or clone the repository and add it to PYTHONPATH."""
    )


from ..data import AtomicData
from ..neighbor_list.neighbor_list import (
    atomic_data2neighbor_list,
    validate_neighborlist,
)


class MACEInterface(torch.nn.Module):
    name: Final[str] = "mace"

    def __init__(self, config: dict, max_num_neighbors: int = 1000):
        super(MACEInterface, self).__init__()
        self.max_num_neighbors = max_num_neighbors
        self.n_atom_types = config["num_elements"]
        for k in "hidden_irreps", "MLP_irreps":
            config[k] = o3.Irreps(config[k])
        for k in "interaction_cls", "interaction_cls_first":
            config[k] = modules.interaction_classes[config[k]]
        # config["atomic_energies"] = np.zeros(self.n_atom_types)
        # config["atomic_numbers"] = [an for an in range(1,self.n_atom_types)]
        config["atomic_energies"] = np.zeros(self.n_atom_types)
        atomic_types = torch.tensor(config["atomic_numbers"])
        self.register_buffer("types_mapping", -1*torch.ones(atomic_types.max()+1,dtype=torch.long))
        self.types_mapping[atomic_types] = torch.arange(atomic_types.shape[0])
        self.model = MACE(**config)
        self.cutoff = config["r_max"]
        self.config = config
        self.derivative = True

    def forward(self, data):
        ndata = self.data2ndata(data)
        with torch.set_grad_enabled(self.derivative):
            out = self.model(ndata, training=self.training)
        data.out[self.name] = {
            "energy": out["energy"].flatten(),
            "forces": out["forces"],
        }
        return data

    def data2ndata(self, data, **kwargs):
        neighbor_list = data.neighbor_list.get(self.name)

        if not self.is_nl_compatible(neighbor_list):
            neighbor_list = self.neighbor_list(
                data, self.cutoff, self.max_num_neighbors
            )[self.name]
        device = data.pos.device
        # one_hot = to_one_hot(data.atom_types.view(-1,1), self.n_atom_types)
        types_ids = self.types_mapping[data.atom_types].view(-1,1)
        one_hot = to_one_hot(types_ids, self.n_atom_types)
        kwargs = dict(
                edge_index=neighbor_list["index_mapping"],
                positions=data.pos,
                node_attrs=one_hot,
                shifts=neighbor_list["cell_shifts"],
                weight=torch.tensor([1], device=device),
        )
        if 'forces' in data:
            kwargs["forces"] = data.forces
        if 'energy' in data:
            kwargs["energy"] = data.energy
        if 'cell' in data:
            if data.cell is not None:
                kwargs["cell"] = data.cell

        ndata = Batch(batch=data.batch, ptr=data.ptr, **kwargs)
        return ndata

    def is_nl_compatible(self, nl):
        is_compatible = False
        if validate_neighborlist(nl):
            if (
                nl["order"] == 2
                and nl["self_interaction"] == False
                and nl["rcut"] == self.cutoff
            ):
                is_compatible = True
        return is_compatible

    @staticmethod
    def neighbor_list(
        data: AtomicData, rcut: float, max_num_neighbors: int = 1000
    ) -> dict:
        """Computes the neighborlist for :obj:`data` using a strict cutoff of :obj:`rcut`."""
        return {
            MACEInterface.name: atomic_data2neighbor_list(
                data,
                rcut,
                self_interaction=False,
                max_num_neighbors=max_num_neighbors,
            )
        }
