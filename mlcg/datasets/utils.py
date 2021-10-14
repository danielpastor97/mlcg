import torch

def remove_baseline_forces(data, models):
    baseline_forces = []
    for k in models.keys():
        models[k].eval()

        data = models[k](data)
        baseline_forces.append(data.out["contributions"][k]["forces"].flatten())
    baseline_forces = torch.sum(torch.vstack(baseline_forces), dim=0).view(
        -1, 3
    )
    data.forces -= baseline_forces
    data.baseline_forces = baseline_forces

    return data
