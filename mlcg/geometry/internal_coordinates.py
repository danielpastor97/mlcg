"""TODO: write tests
"""

import math
import torch
from typing import List, Optional


@torch.jit.script
def safe_norm(
    input: torch.Tensor,
    dim: Optional[List[int]] = None,
    keepdims: bool = True,
    eps: float = 1e-16,
) -> torch.Tensor:
    """Compute Euclidean norm of input so that 0-norm vectors can be used in
    the backpropagation"""
    if dim is None:
        dim = [0]
    return torch.sqrt(
        torch.square(input).sum(dim=dim, keepdim=keepdims) + eps
    ) - math.sqrt(eps)


@torch.jit.script
def safe_normalization(
    input: torch.Tensor, norms: torch.Tensor
) -> torch.Tensor:
    """Normalizes input using norms avoiding divitions by zero"""
    mask = (norms > 0.0).flatten()
    out = input.clone()
    # out = torch.zeros_like(input)
    out[mask] = input[mask] / norms[mask]
    return out


@torch.jit.script
def compute_distance_vectors(
    pos: torch.Tensor,
    mapping: torch.Tensor,
    cell_shifts: Optional[torch.Tensor] = None,
):
    r"""Compute the distance (or displacement) vectors between the positions in
    :obj:`pos` following the :obj:`mapping` assuming that that mapping indices follow::

     i--j

    such that:

    .. math::

        r_{ij} &= ||\mathbf{r}_j - \mathbf{r}_i||_{2} \\
        \hat{\mathbf{r}}_{ij} &= \frac{\mathbf{r}_{ij}}{r_{ij}}

    In the case of periodic boundary conditions, :obj:`cell_shifts` must be
    provided so that :math:`\mathbf{r}_j` can be outside of the original unit
    cell.
    """
    assert mapping.dim() == 2
    assert mapping.shape[0] == 2

    if cell_shifts is None:
        dr = pos[mapping[1]] - pos[mapping[0]]
    else:
        dr = pos[mapping[1]] - pos[mapping[0]] + cell_shifts

    distances = safe_norm(dr, dim=[1])

    direction_vectors = safe_normalization(dr, distances)
    return distances, direction_vectors


@torch.jit.script
def compute_distances(
    pos: torch.Tensor,
    mapping: torch.Tensor,
    cell_shifts: Optional[torch.Tensor] = None,
):
    r"""Compute the distance between the positions in :obj:`pos` following the
    :obj:`mapping` assuming that mapping indices follow::

     i--j

    such that:

    .. math::

        r_{ij} = ||\mathbf{r}_j - \mathbf{r}_i||_{2}

    In the case of periodic boundary conditions, :obj:`cell_shifts` must be
    provided so that :math:`\mathbf{r}_j` can be outside of the original unit
    cell.
    """
    assert mapping.dim() == 2
    assert mapping.shape[0] == 2

    if cell_shifts is None:
        dr = pos[mapping[1]] - pos[mapping[0]]
    else:
        dr = pos[mapping[1]] - pos[mapping[0]] + cell_shifts

    return dr.norm(p=2, dim=1)


@torch.jit.script
def compute_angles(
    pos: torch.Tensor,
    mapping: torch.Tensor,
    cell_shifts: Optional[torch.Tensor] = None,
):
    r"""Compute the cosine of the angle between the positions in :obj:`pos` following the :obj:`mapping` assuming that the mapping indices follow::

       j--k
      /
     i

    .. math::

        \cos{\theta_{ijk}} &= \frac{\mathbf{r}_{ji} \mathbf{r}_{jk}}{r_{ji} r_{jk}}  \\
        r_{ji}&= ||\mathbf{r}_i - \mathbf{r}_j||_{2} \\
        r_{jk}&= ||\mathbf{r}_k - \mathbf{r}_j||_{2}

    In the case of periodic boundary conditions, :obj:`cell_shifts` must be
    provided so that :math:`\mathbf{r}_j` can be outside of the original unit
    cell.
    """
    assert mapping.dim() == 2
    assert mapping.shape[0] == 3

    dr1 = pos[mapping[0]] - pos[mapping[1]]
    dr2 = pos[mapping[2]] - pos[mapping[1]]
    cos_theta = (
        (dr1 * dr2).sum(dim=1) / dr1.norm(p=2, dim=1) / dr2.norm(p=2, dim=1)
    )
    return cos_theta


@torch.jit.script
def compute_dihedrals(pos: torch.Tensor, mapping: torch.Tensor):
    """
    Compute the dihedral angle between positions in :obj:'pos' following the
    :obj:`mapping` assuming that mapping indices follow::

       j--k--l
      /
     i
    Convention is to assign rotations w.r.t position of i&l as positive is l is rotated counterclockwise
    when starting down bond jk

    In the case of periodic boundary conditions, :obj:`cell_shifts` must be
    provided so that :math:`\mathbf{r}_j` can be outside of the original unit
    cell.
    """
    assert mapping.dim() == 2
    assert mapping.shape[0] == 4
    dr1 = pos[mapping[1]] - pos[mapping[0]]
    dr1 = dr1 / dr1.norm(p=2, dim=1)[:, None]
    dr2 = pos[mapping[2]] - pos[mapping[1]]
    dr2 = dr2 / dr2.norm(p=2, dim=1)[:, None]
    dr3 = pos[mapping[3]] - pos[mapping[2]]
    dr3 = dr3 / dr3.norm(p=2, dim=1)[:, None]

    n1 = torch.cross(dr1, dr2, dim=1)
    n2 = torch.cross(dr2, dr3, dim=1)
    m1 = torch.cross(n1, dr2, dim=1)
    y = torch.sum(m1 * n2, dim=-1)
    x = torch.sum(n1 * n2, dim=-1)
    theta = torch.atan2(y, x)

    return theta


@torch.jit.script
def compute_impropers(pos: torch.Tensor, mapping: torch.Tensor):
    """
    Compute the improper angle between positions in :obj:'pos' following the
    :obj:`mapping` assuming that mapping indices follow::
     k
      \
       l--j
      /
     i
    Convention is to assign mapping so l is always the central atom. 
    Finds the angle between planes ikl and ikj

    In the case of periodic boundary conditions, :obj:`cell_shifts` must be
    provided so that :math:`\mathbf{r}_j` can be outside of the original unit
    cell.
    """
    assert mapping.dim() == 2
    assert mapping.shape[0] == 4
    il = pos[mapping[0]] - pos[mapping[3]]
    kl = pos[mapping[1]] - pos[mapping[3]]
    ij = pos[mapping[0]] - pos[mapping[2]]
    kj = pos[mapping[1]] - pos[mapping[2]]

    il = il / il.norm(p=2, dim=1)[:, None]
    kl = kl / kl.norm(p=2, dim=1)[:, None]
    ij = ij / ij.norm(p=2, dim=1)[:, None]
    kj = kj / kj.norm(p=2, dim=1)[:, None]

    planeikl = torch.cross(il, kl, dim=1)
    planeikj = torch.cross(ij, kj, dim=1)
    # Dot product of each row
    costheta = torch.sum(planeikl * planeikj, dim=1)
    costheta = (
        costheta
        / (
            torch.norm(planeikl, p=2, dim=1)[:, None]
            * torch.norm(planeikj, p=2, dim=1)[:, None]
        ).T
    )
    theta = torch.acos(costheta)
    return theta
