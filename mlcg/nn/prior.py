import torch
from torch_scatter import scatter
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
from typing import Final

from ..geometry.topology import Topology
from ..geometry.internal_coordinates import (
    compute_distances,
    compute_angles,
    compute_dihedrals,
)


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
        keys = torch.tensor(list(statistics.keys()), dtype=torch.long)
        self.allowed_interaction_keys = list(statistics.keys())
        self.name = name
        self.order = Harmonic._order_map[name]

        unique_types = torch.unique(keys.flatten())
        assert unique_types.min() >= 0
        max_type = unique_types.max()
        sizes = tuple([max_type + 1 for _ in range(self.order)])
        x_0 = torch.zeros(sizes)
        k = torch.zeros(sizes)
        for key in statistics.keys():
            x_0[key] = statistics[key]["x_0"]
            k[key] = statistics[key]["k"]

        self.register_buffer("x_0", x_0)
        self.register_buffer("k", k)

    def data2features(self, data):
        mapping = data.neighbor_list[self.name]["index_mapping"]
        return Harmonic.compute_features(data.pos, mapping, self.name)

    def forward(self, data):
        mapping = data.neighbor_list[self.name]["index_mapping"]
        mapping_batch = data.neighbor_list[self.name]["mapping_batch"]
        interaction_types = [
            data.atom_types[mapping[ii]] for ii in range(self.order)
        ]
        features = self.data2features(data).flatten()
        y = Harmonic.compute(
            features, self.x_0[interaction_types], self.k[interaction_types], 0
        )
        y = scatter(y, mapping_batch, dim=0, reduce="sum")
        data.out[self.name] = {"energy": y}
        return data

    @staticmethod
    def compute_features(pos, mapping, target):
        return Harmonic._compute_map[target](pos, mapping)

    @staticmethod
    def compute(x, x0, k, V0):
        return k * (x - x0) ** 2 + V0

    @staticmethod
    def fit_from_potential_estimates(bin_centers_nz, dG_nz):
        # remove noise by discarding signals
        integral = torch.tensor(
            float(trapezoid(dG_nz.cpu().numpy(), bin_centers_nz.cpu().numpy()))
        )

        mask = torch.abs(dG_nz) > 1e-4 * torch.abs(integral)
        try:
            popt, _ = curve_fit(
                Harmonic.compute,
                bin_centers_nz[mask],
                dG_nz[mask],
                p0=[bin_centers_nz[torch.argmin(dG_nz[mask])], 60, -1],
            )
            stat = {"k": popt[1], "x_0": popt[0]}
        except:
            print(f"failed to fit potential estimate for Harmonic")
            stat = {
                "k": torch.tensor(float("nan")),
                "x_0": torch.tensor(float("nan")),
            }
        return stat

    @staticmethod
    def neighbor_list(topology: Topology, type: str) -> None:
        assert type in Harmonic._compute_map
        nl = topology.neighbor_list(type)
        return {type: nl}


class HarmonicBonds(Harmonic):
    name: Final[str] = "bonds"
    _order = 2

    def __init__(self, statistics) -> None:
        super(HarmonicBonds, self).__init__(statistics, HarmonicBonds.name)

    @staticmethod
    def neighbor_list(topology: Topology) -> dict:
        return Harmonic.neighbor_list(topology, HarmonicBonds.name)

    @staticmethod
    def compute_features(pos, mapping):
        return Harmonic.compute_features(pos, mapping, HarmonicBonds.name)


class HarmonicAngles(Harmonic):
    name: Final[str] = "angles"
    _order = 2

    def __init__(self, statistics) -> None:
        super(HarmonicAngles, self).__init__(statistics, HarmonicAngles.name)

    @staticmethod
    def neighbor_list(topology: Topology) -> dict:
        return Harmonic.neighbor_list(topology, HarmonicAngles.name)

    @staticmethod
    def compute_features(pos, mapping):
        return Harmonic.compute_features(pos, mapping, HarmonicAngles.name)


class Repulsion(torch.nn.Module, _Prior):
    name: Final[str] = "repulsion"
    _neighbor_list_name = "fully connected"

    def __init__(self, statistics) -> None:
        super(Repulsion, self).__init__()
        keys = torch.tensor(list(statistics.keys()), dtype=torch.long)
        self.allowed_interaction_keys = list(statistics.keys())
        self.order = 2
        self.name = self.name
        unique_types = torch.unique(keys.flatten())
        assert unique_types.min() >= 0
        max_type = unique_types.max()
        sizes = tuple([max_type + 1 for _ in range(self.order)])
        sigma = torch.zeros(sizes)
        for key in statistics.keys():
            sigma[key] = statistics[key]["sigma"]
        self.register_buffer("sigma", sigma)

    def data2features(self, data):
        mapping = data.neighbor_list[self.name]["index_mapping"]
        return Repulsion.compute_features(data.pos, mapping)

    def forward(self, data):
        mapping = data.neighbor_list[self.name]["index_mapping"]
        mapping_batch = data.neighbor_list[self.name]["mapping_batch"]
        interaction_types = [
            data.atom_types[mapping[ii]] for ii in range(self.order)
        ]
        features = self.data2features(data)
        y = Repulsion.compute(features, self.sigma[interaction_types])
        y = scatter(y, mapping_batch, dim=0, reduce="sum")
        data.out[self.name] = {"energy": y}
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
    def neighbor_list(topology: Topology) -> dict:
        return {
            Repulsion.name: topology.neighbor_list(
                Repulsion._neighbor_list_name
            )
        }


class Dihedral(torch.nn.Module, _Prior):
    """
    TO DO: better guess for p0 under fit_from_potential_estimates
    """

    _name : Final[str] = "dihedral"
    _order = 4
    _neighbor_list_name = "dihedrals"

    def __init__(self, statistics) -> None:
        super(Dihedral, self).__init__()
        keys = torch.tensor(list(statistics.keys()), dtype=torch.long)
        self.allowed_interaction_keys = list(statistics.keys())
        self.order = self._order
        self.name = self._name
        unique_types = torch.unique(keys.flatten())
        assert unique_types.min() >= 0
        max_type = unique_types.max()
        sizes = tuple([max_type + 1 for _ in range(self.order)])
        theta_0 = torch.zeros(sizes)
        k_0 = torch.zeros(sizes)
        theta_1 = torch.zeros(sizes)
        k_1 = torch.zeros(sizes)
        theta_2 = torch.zeros(sizes)
        k_2 = torch.zeros(sizes)

        for key in statistics.keys():
            theta_0[key] = statistics[key]["theta_0"]
            k_0[key] = statistics[key]["k_0"]
            theta_1[key] = statistics[key]["theta_1"]
            k_1[key] = statistics[key]["k_1"]
            theta_2[key] = statistics[key]["theta_2"]
            k_2[key] = statistics[key]["k_2"]

        self.register_buffer("theta_0", theta_0)
        self.register_buffer("k_0", k_0)
        self.register_buffer("theta_1", theta_1)
        self.register_buffer("k_1", k_1)
        self.register_buffer("theta_2", theta_1)
        self.register_buffer("k_2", k_1)

    def data2features(self, data):
        mapping = data.neighbor_list[self.name]["index_mapping"]
        return Dihedral.compute_features(data.pos, mapping)

    def forward(self, data):
        mapping = data.neighbor_list[self.name]["index_mapping"]
        mapping_batch = data.neighbor_list[self.name]["mapping_batch"]
        interaction_types = [
            data.atom_types[mapping[ii]] for ii in range(self.order)
        ]
        features = self.data2features(data).flatten()
        y = Dihedral.compute(
            features,
            self.theta_0[interaction_types],
            self.k_0[interaction_types],
            self.theta_1[interaction_types],
            self.k_1[interaction_types],
            self.theta_2[interaction_types],
            self.k_2[interaction_types],
        )
        y = scatter(y, mapping_batch, dim=0, reduce="sum")
        data.out[self.name] = {"energy": y}
        return data

    @staticmethod
    def compute_features(pos, mapping):
        return compute_dihedrals(pos, mapping)

    @staticmethod
    def compute1(theta, theta_0, k_0):
        V = 0
        V += k_0 * (1 - torch.cos(1 * theta - theta_0))
        return V

    @staticmethod
    def compute2(theta, theta_0, k_0, theta_1, k_1):
        V = 0
        V += k_0 * (1 - torch.cos(1 * theta - theta_0))
        V += k_1 * (1 - torch.cos(2 * theta - theta_1))
        return V

    @staticmethod
    def compute3(theta, theta_0, k_0, theta_1, k_1, theta_2, k_2):
        V = 0
        V += k_0 * (1 - torch.cos(1 * theta - theta_0))
        V += k_1 * (1 - torch.cos(2 * theta - theta_1))
        V += k_2 * (1 - torch.cos(3 * theta - theta_2))
        return V

    @staticmethod
    def neg_log_likelihood(y, yhat):
        """
        Convert dG to probability and use KL divergence to get difference between
        predicted and actual
        """
        L = torch.sum(
            torch.exp(-y) * torch.log(torch.exp(-y) / torch.exp(-yhat))
        )
        return -L

    @staticmethod
    def fit_from_potential_estimates(bin_centers_nz, dG_nz):
        integral = torch.tensor(
            float(trapezoid(dG_nz.cpu().numpy(), bin_centers_nz.cpu().numpy()))
        )

        mask = torch.abs(dG_nz) > 1e-4 * torch.abs(integral)
        try:
            # Determine best fit. Either 1, 2, or 3 parameters
            compute_dihedrals = [
                Dihedral.compute1,
                Dihedral.compute2,
                Dihedral.compute3,
            ]
            free_parameters = [2, 4, 6]
            p0s = [[3.1415 / 2, 1], [3.1415 / 2, 1, -3.1415 / 2, 1]]
            popts = []
            aics = []
            for i_cd, compute_dihedral in enumerate(compute_dihedrals):
                popt, _ = curve_fit(
                    compute_dihedral,
                    bin_centers_nz[mask],
                    dG_nz[mask],
                    p0s[i_cd],
                )
                popts.append(popt)
                aic = (
                    2
                    * Dihedral.neg_log_likelihood(
                        dG_nz[mask],
                        compute_dihedral(bin_centers_nz[mask], *popt),
                    )
                    - 2 * free_parameters[i_cd]
                )
                aics.append(aic)

            min_aic = min(aics)
            min_i_aic = aics.index(min_aic)

            if min_i_aic == 0:
                popt = popts[0]
                stat = {
                    "theta_1": 0,
                    "k_1": 0,
                    "theta_2": 0,
                    "k_2": 0,
                }
                Dihedral.compute = Dihedral.compute1
            elif min_i_aic == 1:
                popt = popts[1]
                stat = {
                    "theta_1": popt[2],
                    "k_1": popt[3],
                    "theta_2": 0,
                    "k_2": 0,
                }
                Dihedral.compute = Dihedral.compute2
            elif min_i_aic == 2:
                popt = popts[2]
                stat = {
                    "theta_1": popt[2],
                    "k_1": popt[3],
                    "theta_2": popt[4],
                    "k_2": popt[5],
                }
                Dihedral.compute = Dihedral.compute3
            stat = {"theta_0": popt[0], "k_0": popt[1]}

        except:
            print(f"failed to fit potential estimate for Dihedral")
            stat = {
                "theta_0": torch.tensor(float("nan")),
                "k_0": torch.tensor(float("nan")),
                "theta_1": torch.tensor(float("nan")),
                "k_1": torch.tensor(float("nan")),
                "theta_2": torch.tensor(float("nan")),
                "k_2": torch.tensor(float("nan")),
            }
        return stat

    def from_user(*args):
        """
        Direct input of parameters from user
        """
        stat = {
            "theta_0": args[0],
            "k_0": args[1],
            "theta_1": args[2],
            "k_1": args[3],
            "theta_2": args[4],
            "k_2": args[5],
        }
        return stat

    @staticmethod
    def neighbor_list(topology) -> None:
        nl = topology.neighbor_list("dihedrals")
        return {Dihedral._name: nl}
