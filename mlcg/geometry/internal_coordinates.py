import torch

from .math_utils import safe_norm, safe_normalization

@torch.jit.script
def compute_distance_vectors(pos:torch.Tensor, mapping:torch.Tensor):
    assert mapping.dim() == 2
    assert mapping.shape[1] == 2

    dr = pos[mapping[1]] - pos[mapping[0]]

    distances = safe_norm(dr, dim=1)

    direction_vectors = safe_normalization(dr, distances)
    return distances, direction_vectors

@torch.jit.script
def compute_bound_distances(pos:torch.Tensor, mapping:torch.Tensor):
    assert mapping.dim() == 2
    assert mapping.shape[1] == 2

    dr = pos[mapping[1]] - pos[mapping[0]]
    distances = safe_norm(dr, dim=1)
    return distances

@torch.jit.script
def compute_bound_angles(pos:torch.Tensor, mapping:torch.Tensor):
    assert mapping.dim() == 2
    assert mapping.shape[1] == 3

    dr1 = pos[mapping[1]] - pos[mapping[0]]
    dr2 = pos[mapping[2]] - pos[mapping[0]]
    cos_theta = dr1 @ dr2.t()
    return cos_theta