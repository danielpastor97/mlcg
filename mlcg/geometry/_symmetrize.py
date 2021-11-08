import torch


def _symmetrise_angle_interaction(
    unique_interaction_types: torch.tensor,
) -> torch.tensor:
    """For angles defined as::

      2---3
     /
    1

    atom 1 and 3 can be exchanged without changing the angle so the resulting
    interaction is symmetric w.r.t such transformations. Hence the need for only
    considering interactions (a,b,c) with a < c.

    """
    mask = unique_interaction_types[0] > unique_interaction_types[2]
    ee = unique_interaction_types[0, mask]
    unique_interaction_types[0, mask] = unique_interaction_types[2, mask]
    unique_interaction_types[2, mask] = ee
    return unique_interaction_types


def _symmetrise_distance_interaction(
    unique_interaction_types: torch.tensor,
) -> torch.tensor:
    """Distance based interactions are symmetric w.r.t. the direction hence
    the need for only considering interactions (a,b) with a < b.
    """
    mask = unique_interaction_types[0] > unique_interaction_types[1]
    ee = unique_interaction_types[0, mask]
    unique_interaction_types[0, mask] = unique_interaction_types[1, mask]
    unique_interaction_types[1, mask] = ee
    return unique_interaction_types


_symmetrise_map = {
    2: _symmetrise_distance_interaction,
    3: _symmetrise_angle_interaction,
}

_flip_map = {
    2: lambda tup: torch.tensor([tup[1], tup[0]], dtype=torch.long),
    3: lambda tup: torch.tensor([tup[2], tup[1], tup[0]], dtype=torch.long),
}
