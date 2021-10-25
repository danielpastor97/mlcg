from typing import Dict
from copy import deepcopy
import torch
from scipy.integrate import trapezoid

from ..data import AtomicData
from ..nn.prior import Harmonic, _Prior
from ..utils import tensor2tuple


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


def _get_all_unique_keys(
    unique_types: torch.tensor, order: int
) -> torch.tensor:
    """Helper function for returning all unique, symmetrised atom type keys

    Parameters
    ----------
    unique_types:
        Tensor of unique atom types of shape (order, n_unique_atom_types)
    order:
        The order of the interaction type

    Returns
    -------
    unique_sym_keys:
       Tensor of unique atom types, symmetrised
    """
    # get all combinations of size order between the elements of unique_types
    keys = torch.cartesian_prod(*[unique_types for ii in range(order)]).t()
    # symmetrize the keys and keep only unique entries
    sym_keys = _symmetrise_map[order](keys)
    unique_sym_keys = torch.unique(sym_keys, dim=1)
    return unique_sym_keys


def _get_bin_centers(
    feature: torch.tensor, nbins: int, amin: float = None, amax: float = None
) -> torch.tensor:
    """Returns bin centers for histograms.

    Parameters
    ----------
    feature:
        1-D input values of a feature.
    nbins:
        Number of bins in the histogram
    amin
        If specified, the lower bound of bin edges. If not specified, the lower bound
        defaults to the lowest value in the input feature
    amax
        If specified, the upper bound of bin edges. If not specified, the upper bound
        defaults to the greatest value in the input feature

    Returns
    -------
    bin_centers:
        The locaations of the bin centers
    """
    if amin != None and amax != None:
        if amin >= amax:
            raise ValueError("amin must be less than amax.")

    bin_centers = torch.zeros((nbins,), dtype=torch.float64)
    if amin != None:
        a_min = amin
    else:
        a_min = a.min()
    if a_max != None:
        a_max = amax
    else:
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
    amin: float = None,
    amax: float = None,
) -> Dict:
    r"""Function for computing atom type-specific statistics for
    every combination of atom types present in a collated AtomicData
    structure.

    Parameters
    ----------
    data:
        Input data, in the form of a collated list of individual AtomicData
        structures.
    target:
        The keyword specifiying with `neighbor_list` sub_dictionary should
        be used to gather statisitics
    beta:
        Inverse thermodynamic temperature:

        .. math::

            \beta = \frac{1}{k_B T}

        where :math:`k_B` is Boltzmann's constant and :math:`T` is the temperature.
    TargetPrior:
        The class type of prior for which the statistics will be processed.
    nbins:
        The number of bins over which 1-D feature histograms are constructed
        in order to estimate distributions
    amin
        If specified, the lower bound of bin edges. If not specified, the lower bound
        defaults to the lowest value in the input feature
    amax
        If specified, the upper bound of bin edges. If not specified, the upper bound
        defaults to the greatest value in the input feature

    Returns
    -------
    statistics:
        Dictionary of gathered statistics and estimated parameters based on
        the `TargetPrior`. The following key/value pairs are common across all `TargetPrior`
        choices:

            (*specific_types) : {
                "p" : torch.tensor of shape [n_bins], containing the normalized bin counts
                    of the of the 1-D feature corresponding to the atom_type group
                    (*specific_types) = (specific_types[0], specific_types[1], ...)
                "p_bin: : torch.tensor of shape [n_bins] containing the bin center values
                "V" : torch.tensor of shape [n_bins], containing the emperically estimated
                    free energy curve according to a directly Boltzmann inversion:

                        .. math::

                            V = -\frac{1}{\beta}\log{\left( p \right)}

                "V_bin" : torch_tensor of shape [n_bins], containing the bin center values

        Other sub-key/value pairs apart from those enumerated above, may appear depending
        on the chosen `TargetPrior`. For example, if `TargetPrior` is `HarmonicBonds`, there
        will also be keys/values associated with estimated bond constants and means.

    Example
    -------
    ```
    my_data = AtomicData(
        out={},
        pos=[769600, 3],
        atom_types=[769600],
        n_atoms=[20800],
        neighbor_list={
            bonds={
              tag=[20800],
              order=[20800],
              index_mapping=[2, 748800],
              cell_shifts=[20800],
              rcut=[20800],
              self_interaction=[20800]
            },
            angles={
              tag=[20800],
              order=[20800],
              index_mapping=[3, 977600],
              cell_shifts=[20800],
              rcut=[20800],
              self_interaction=[20800]
            }
        },
        batch=[769600],
        ptr=[20801]
    )

    angle_stats = bond_stats = compute_statistics(my_data,
         'bonds', beta=beta,
         TargetPrior=HarmonicBonds
    )
    ```

    """

    unique_types = torch.unique(data.atom_types)
    order = data.neighbor_list[target]["index_mapping"].shape[0]
    unique_keys = _get_all_unique_keys(unique_types, order)

    mapping = data.neighbor_list[target]["index_mapping"]
    values = TargetPrior.compute_features(data.pos, mapping)

    interaction_types = torch.vstack(
        [data.atom_types[mapping[ii]] for ii in range(order)]
    )

    interaction_types = _symmetrise_map[order](intereaction_types)

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

        bin_centers = _get_bin_centers(val, nbins, amin=amin, amax=amax)
        hist = torch.histc(val, bins=nbins, min=amin, max=amax)

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

        kf = tensor2tuple(_flip_map[order](unique_key))
        statistics[kf] = deepcopy(statistics[kk])

    return statistics


def fit_baseline_models(
    data: AtomicData,
    beta: float,
    priors_cls: List[_Prior],
    nbins: int = 100,
    amin: float = None,
    amax: float = None,
) -> Tuple[List[nn.Module], Dict]:
    """Function for parametrizing a list of priors based on type-specific interactions contained in
    a collated AtomicData structure

    Parameters
    ----------
    data:
        Input data, in the form of a collated list of individual AtomicData
        structures.
    beta:
        Inverse thermodynamic temperature:

        .. math::

        \beta = \frac{1}{k_B T}

        where :math:`k_B` is Boltzmann's constant and :math:`T` is the temperature.
    priors_cls:
       List of priors to parametrize based on the input data
    nbins:
        The number of bins over which 1-D feature histograms are constructed
        in order to estimate distributions
    amin
        If specified, the lower bound of bin edges. If not specified, the lower bound
        defaults to the lowest value in the input feature
    amax
        If specified, the upper bound of bin edges. If not specified, the upper bound
        defaults to the greatest value in the input feature

    Returns
    -------
    models, statistics:
        The list of parametrized priors and a dictionary containing their coresponding statistics;
        see `compute_statistics` for more detailed information.
    """

    statistics = {}
    models = torch.nn.ModuleDict()
    for TargetPrior in priors_cls:
        k = TargetPrior._name
        statistics[k] = compute_statistics(
            data,
            k,
            beta,
            TargetPrior=TargetPrior,
            nbins=nbins,
            amin=amin,
            amax=amax,
        )
        models[k] = TargetPrior(statistics[k])
    return models, statistics
