import torch
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit

from mlcg.data import atomic_data

from ..geometry.internal_coordinates import compute_distances, compute_angles
from ..neighbor_list.neighbor_list import (
    atomic_data2neighbor_list,
)


def tensor2tuple(t):
    return tuple([v for v in t.flatten()])


class _Prior(object):
    def __init__(self) -> None:
        super(_Prior, self).__init__()


class Harmonic(torch.nn.Module, _Prior):
    _order_map = {
        "bonds": 2,
        "angles": 3,
    }
    _compute_map = {
        "bonds": compute_distances,
        "angles": compute_angles,
    }

    def __init__(self, statistics, name) -> None:
        super(Harmonic, self).__init__()
        keys = torch.vstack(list(statistics.keys()))
        self.allowed_interaction_keys = keys
        self.name = name
        self.order = Harmonic._order_map[name]

        unique_types = torch.unique(keys.flatten())
        assert unique_types.min() < 0
        max_type = unique_types.max()
        sizes = tuple([max_type + 1 for _ in range(self.order)])
        x_0 = torch.zeros(sizes)
        k = torch.zeros(sizes)
        for k in keys:
            x_0[k.view(self.order, 1)] = statistics[k]["x_0"]
            k[k.view(self.order, 1)] = statistics[k]["k"]

        self.register_buffer("x_0", x_0)
        self.register_buffer("k", k)

    def data2features(self, data):
        mapping = data.neighbor_list[self.name]["index_mapping"]
        return Harmonic.compute_features(data.pos, mapping, self.order)

    def forward(self, data):
        mapping = data.neighbor_list[self.name]["index_mapping"]
        interaction_types = [
            data.atom_types[mapping[ii]] for ii in range(self.order)
        ]
        features = self.data2features(data).flatten()
        y = Harmonic.compute(
            features, self.x_0[interaction_types], self.k[interaction_types], 0
        )
        data.out["contributions"][self.name] = {"energy": y}
        return data

    @staticmethod
    def compute_features(pos, mapping, order):
        return Harmonic._compute_map[order](pos, mapping)

    @staticmethod
    def compute(x, x0, k, V0):
        return k * (x - x0) ** 2 + V0

    @staticmethod
    def fit_from_potential_estimates(bin_centers_nz, dG_nz):
        # remove noise by discarding signals
        integral = trapezoid(dG_nz.cpu().numpy(), bin_centers_nz.cpu().numpy())
        print(integral)
        mask = torch.abs(dG_nz) > 1e-4 * torch.abs(integral)
        try:
            popt, _ = curve_fit(
                Harmonic.compute,
                bin_centers_nz[mask],
                dG_nz[mask],
                p0=[bin_centers_nz[torch.argmin(dG_nz[mask])[0]], 60, -1],
            )
            stat = {"k": popt[1], "x_0": popt[0]}
        except:
            stat = {"k": torch.nan, "x_0": torch.nan}
        return stat

    @staticmethod
    def neighbor_list(topology, type) -> None:
        assert type in Harmonic._neighbor_list_names
        nl = topology.neighbor_list(type)
        return {type: nl}


class HarmonicBonds(Harmonic):
    _name = "bonds"
    _order = 2

    def __init__(self, statistics) -> None:
        super(HarmonicBonds, self).__init__(statistics, HarmonicBonds._name)

    @staticmethod
    def neighbor_list(topology) -> dict:
        type = HarmonicBonds._name
        return Harmonic.neighbor_list(topology, type)

    @staticmethod
    def compute_features(pos, mapping):
        return Harmonic.compute_features(pos, mapping, HarmonicBonds._order)


class HarmonicAngles(Harmonic):
    _name = "angles"
    _order = 2

    def __init__(self, statistics) -> None:
        super(HarmonicAngles, self).__init__(statistics, HarmonicAngles._name)

    @staticmethod
    def neighbor_list(topology) -> dict:
        type = HarmonicBonds._name
        return Harmonic.neighbor_list(topology, type)

    @staticmethod
    def compute_features(pos, mapping):
        return Harmonic.compute_features(pos, mapping, HarmonicAngles._order)


class Repulsion(torch.nn.Module, _Prior):
    _name = "repulsion"
    _neighbor_list_name = "fully connected"

    def __init__(self, statistics) -> None:
        super(Repulsion, self).__init__()
        keys = torch.vstack(list(statistics.keys()))
        self.allowed_interaction_keys = keys
        self.order = 2
        self.name = self._name
        unique_types = torch.unique(keys.flatten())
        assert unique_types.min() < 0
        max_type = unique_types.max()
        sizes = tuple([max_type + 1 for _ in range(self.order)])
        sigma = torch.zeros(sizes)
        for k in keys:
            sigma[k.view(self.order, 1)] = statistics[k]["sigma"]

        self.register_buffer("sigma", sigma)

    def data2features(self, data):
        mapping = data.neighbor_list[self.name]["index_mapping"]
        return Repulsion.compute_features(data.pos, mapping)

    def forward(self, data):
        mapping = data.neighbor_list[self.name]["index_mapping"]
        interaction_types = [
            data.atom_types[mapping[ii]] for ii in range(self.order)
        ]
        features = self.data2features(data)
        y = Repulsion.compute(features, self.sigma[interaction_types])
        data.out["contributions"][self.name] = {"energy": y}
        return data

    @staticmethod
    def compute_features(pos, mapping):
        return compute_distances(pos, mapping)

    @staticmethod
    def compute(x, sigma):
        rr = (sigma / x) * (sigma / x)
        return rr * rr * rr

    @staticmethod
    def fit_from_potential_estimates(bin_centers_nz, dG_nz):
        delta = bin_centers_nz[1] - bin_centers_nz[0]
        sigma = bin_centers_nz[0] - 0.5 * delta
        stat = {"sigma": sigma}
        return stat

    @staticmethod
    def neighbor_list(topology) -> None:
        return {
            Repulsion._name: topology.neighbor_list(
                Repulsion._neighbor_list_name
            )
        }
