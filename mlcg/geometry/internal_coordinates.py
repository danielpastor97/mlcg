import math
import torch
from typing import List,Optional


@torch.jit.script
def safe_norm(input: torch.Tensor, dim:Optional[List[int]]=None, keepdims:bool=True, eps:float=1e-16)->torch.Tensor:
    """Compute Euclidean norm of input so that 0-norm vectors can be used in
    the backpropagation"""
    if dim is None:
        dim = [0]
    return torch.sqrt(
        torch.square(input).sum(dim=dim, keepdim=keepdims) + eps
    ) - math.sqrt(eps)

@torch.jit.script
def safe_normalization(input: torch.Tensor, norms: torch.Tensor)->torch.Tensor:
    """Normalizes input using norms avoiding divitions by zero"""
    mask = (norms > 0.0).flatten()
    out = input.clone()
    # out = torch.zeros_like(input)
    out[mask] = input[mask] / norms[mask]
    return out

@torch.jit.script
def compute_distance_vectors(pos: torch.Tensor, mapping: torch.Tensor):
    assert mapping.dim() == 2
    assert mapping.shape[0] == 2

    dr = pos[mapping[1]] - pos[mapping[0]]

    distances = safe_norm(dr, dim=[1])

    direction_vectors = safe_normalization(dr, distances)
    return distances, direction_vectors


@torch.jit.script
def compute_distances(pos: torch.Tensor, mapping: torch.Tensor):
    assert mapping.dim() == 2
    assert mapping.shape[0] == 2

    dr = pos[mapping[1]] - pos[mapping[0]]
    return dr.norm(p=2,dim=1)


@torch.jit.script
def compute_angles(pos: torch.Tensor, mapping: torch.Tensor):
    assert mapping.dim() == 2
    assert mapping.shape[0] == 3

    dr1 = pos[mapping[0]] - pos[mapping[1]]
    dr2 = pos[mapping[2]] - pos[mapping[1]]
    cos_theta = (dr1 * dr2).sum(dim=1) / dr1.norm(p=2,dim=1) / dr2.norm(p=2,dim=1)
    return cos_theta


