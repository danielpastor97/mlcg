import torch
from itertools import product
import numpy as np

from ..data import AtomicData
from .internal_coordinates import compute_distances ,compute_angles

compute_map = {
    2: compute_distances,
    3: compute_angles,
}

def symmetrise_angle(tup):
    tup = tuple(tup)
    ends = sorted([tup[0], tup[2]])
    return (ends[0], tup[1], ends[1])
def symmetrise_distance(tup):
    tup = sorted(tup)
    return (tup[0], tup[1])

symmetrise_map = {
    2: symmetrise_distance,
    3: symmetrise_angle,
}

def compute_statistics(data: AtomicData, target: str, beta: float):

    unique_types = torch.unique(data.atomic_types)
    order = data.neighbor_list[target]['order']
    keys = set(map(symmetrise_map[order],product(*[unique_types for _ in range(order)])))
    statistics = {k:{'values':[], 'mean':0, 'k':0, 'std':0} for k in keys}
    mapping = data.neighbor_list[target]['index_mapping']
    vars = compute_map[order](data.pos, mapping)

    interaction_types = torch.cat([data.atomic_types[mapping[ii]].unsqueeze(0) for ii in range(order)], dim=0)

    for val, interaction_type in zip(vars,interaction_types):
        statistics[symmetrise_map(interaction_type)].append(val)

    for k in statistics.keys():
        values = statistics[k].pop('values')
        statistics[k]['mean'] = np.mean(values)
        statistics[k]['std'] = np.std(values)
        statistics[k]['k'] = 1/np.var(values)/beta
    return statistics