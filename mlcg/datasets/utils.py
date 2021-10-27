import torch
from torch_geometric.loader import DataLoader
from typing import Dict, List, Sequence
from mlcg.data.atomic_data import AtomicData

from ..data._keys import FORCE_KEY


def chunker(seq: Sequence, size:int):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def remove_baseline_forces(data_list: List[AtomicData], models: Dict[str, torch.nn.Module]):
    """Compute the forces on the input :obj:`data_list` with the :obj:`models`
    and remove them from the reference forces contained in :obj:`data_list`.
    The computation of the forces is done on the whole :obj:`data_list` at once
    so it should not be too large.
    """
    n_frame = len(data_list)
    dataloader = DataLoader(data_list, batch_size=n_frame)
    baseline_forces = []
    for data in dataloader:
        for k in models.keys():
            models[k].eval()

            data = models[k](data)
            baseline_forces.append(data.out[k][FORCE_KEY].flatten())
            # make sure predicted properties don't require gradient anymore
            for key, v in data.out[k].items():
                data.out[k][key] = v.detach()
    baseline_forces = torch.sum(torch.vstack(baseline_forces), dim=0).view(
        -1, 3
    )

    for i_frame in range(n_frame):
        mask = data.batch == i_frame
        data_list[i_frame].forces -= baseline_forces[mask]
        data_list[i_frame].baseline_forces = baseline_forces[mask]

    return data_list
