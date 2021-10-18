import torch

from ..data._keys import FORCE_KEY


def remove_baseline_forces(data, models):
    """Compute the forces on the input :obj:`data` with the :obj:`models` and
    remove them from the reference forces contained in :obj:`data`.
    """
    baseline_forces = []
    for k in models.keys():
        models[k].eval()

        data = models[k](data)
        baseline_forces.append(data.out[k][FORCE_KEY].flatten())
    baseline_forces = torch.sum(torch.vstack(baseline_forces), dim=0).view(
        -1, 3
    )
    data.forces -= baseline_forces
    data.baseline_forces = baseline_forces

    return data
