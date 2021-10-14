from copy import deepcopy
import torch
from scipy.integrate import trapezoid

from ..data import AtomicData
from ..nn.prior import Harmonic, Repulsion, _Prior
from ..utils import tensor2tuple


def symmetrise_angle_interaction(unique_interaction_types):
    mask = unique_interaction_types[0] > unique_interaction_types[2]
    ee = unique_interaction_types[0, mask]
    unique_interaction_types[0, mask] = unique_interaction_types[2, mask]
    unique_interaction_types[2, mask] = ee
    unique_interaction_types = torch.unique(unique_interaction_types, dim=1)
    return unique_interaction_types


def symmetrise_distance_interaction(unique_interaction_types):
    mask = unique_interaction_types[0] > unique_interaction_types[1]
    ee = unique_interaction_types[0, mask]
    unique_interaction_types[0, mask] = unique_interaction_types[1, mask]
    unique_interaction_types[1, mask] = ee
    unique_interaction_types = torch.unique(unique_interaction_types, dim=1)
    return unique_interaction_types


symmetrise_map = {
    2: symmetrise_distance_interaction,
    3: symmetrise_angle_interaction,
}
flip_map = {
    2: lambda tup: torch.tensor([tup[1], tup[0]], dtype=torch.long),
    3: lambda tup: torch.tensor([tup[2], tup[1], tup[0]], dtype=torch.long),
}


def get_all_unique_keys(unique_types, order):
    # get all combinations of size order between the elements of unique_types
    keys = torch.cartesian_prod(*[unique_types for ii in range(order)]).t()
    # symmetrize the keys and keep only unique entries
    sym_keys = symmetrise_map[order](keys)
    unique_sym_keys = torch.unique(sym_keys, dim=1)
    return unique_sym_keys


def get_bin_centers(a, nbins):
    bin_centers = torch.zeros((nbins,), dtype=torch.float64)
    a_min = a.min()
    a_max = a.max()
    delta = (a_max - a_min) / nbins
    bin_centers = (
        a_min
        + 0.5 * delta
        + torch.arange(0, nbins, dtype=torch.float64) * delta
    )
    return bin_centers


def compute_statistics(
    data: AtomicData,
    target: str,
    beta: float,
    TargetPrior: _Prior = Harmonic,
    nbins: int = 100,
):

    unique_types = torch.unique(data.atom_types)
    order = data.neighbor_list[target]["index_mapping"].shape[0]
    unique_keys = get_all_unique_keys(unique_types, order)

    mapping = data.neighbor_list[target]["index_mapping"]
    values = TargetPrior.compute_features(data.pos, mapping)

    interaction_types = torch.vstack(
        [data.atom_types[mapping[ii]] for ii in range(order)]
    )

    statistics = {}
    for unique_key in unique_keys.t():
        # find which values correspond to unique_key type of interaction
        mask = torch.all(
            torch.vstack(
                [
                    interaction_types[ii, :] == unique_key[ii]
                    for ii in range(order)
                ]
            ),
            dim=0,
        )
        val = values[mask]
        if len(val) == 0:
            continue

        bin_centers = get_bin_centers(val, nbins)
        hist = torch.histc(val, bins=nbins)

        mask = hist > 0
        bin_centers_nz = bin_centers[mask]
        ncounts_nz = hist[mask]
        dG_nz = -torch.log(ncounts_nz) / beta
        params = TargetPrior.fit_from_potential_estimates(bin_centers_nz, dG_nz)
        kk = tensor2tuple(unique_key)
        statistics[kk] = params

        statistics[kk]["p"] = hist / trapezoid(
            hist.cpu().numpy(), x=bin_centers.cpu().numpy()
        )
        statistics[kk]["p_bin"] = bin_centers
        statistics[kk]["V"] = dG_nz
        statistics[kk]["V_bin"] = bin_centers_nz

        kf = tensor2tuple(flip_map[order](unique_key))
        statistics[kf] = deepcopy(statistics[kk])

    return statistics


def fit_baseline_models(data, beta, priors_cls, nbins: int = 100):
    statistics = {}
    models = torch.nn.ModuleDict()
    for TargetPrior in priors_cls:
        k = TargetPrior._name
        statistics[k] = compute_statistics(
            data, k, beta, TargetPrior=TargetPrior, nbins=nbins
        )
        models[k] = TargetPrior(statistics[k])
    return models, statistics
