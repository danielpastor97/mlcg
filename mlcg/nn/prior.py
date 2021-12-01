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

    name: Final[str] = "dihedrals"
    _order = 4
    _neighbor_list_name = "dihedrals"

    def __init__(self, statistics) -> None:
        super(Dihedral, self).__init__()
        keys = torch.tensor(list(statistics.keys()), dtype=torch.long)
        self.allowed_interaction_keys = list(statistics.keys())
        self.order = self._order
        self.name = self.name
        unique_types = torch.unique(keys.flatten())
        assert unique_types.min() >= 0
        max_type = unique_types.max()
        sizes = tuple([max_type + 1 for _ in range(self.order)])
        # In principle we could extend this to include even more wells if needed.
        theta = 3*[torch.zeros(sizes)]
        self.theta_names = ["theta_"+str(ii) for ii in range(len(theta))]
        k = 3*[torch.zeros(sizes)]
        self.k_names = ["k_"+str(ii) for ii in range(len(k))]

        for key in statistics.keys():
            for ii in range(len(theta)):
                theta_name = self.theta_names[ii]
                k_name = self.k_names[ii]
                theta[ii][key] = statistics[key][theta_name]
                k[ii][key] = statistics[key][k_name]
        self.register_buffer("thetas", theta)
        self.register_buffer("ks", k)

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
            self.thetas[interaction_types],
            self.ks[interaction_types],
        )
        y = scatter(y, mapping_batch, dim=0, reduce="sum")
        data.out[self.name] = {"energy": y}
        return data

    @staticmethod
    def compute_features(pos, mapping):
        return compute_dihedrals(pos, mapping)


    class ComputeClass:
        def __init__(self):
            pass
        def compute(self,theta,theta0s,ks,deg):
            V = torch.zeros_like(theta)
            theta0s[:(deg+1)] = 0
            ks[:(deg+1)] = 0
            for ii,(theta0, k) in enumerate(zip(theta0s, ks)):
                ii = torch.Tensor(ii+1)
                V += k * (1 - torch.cos(ii*theta - theta0))
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
        stat = {}
        integral = torch.tensor(
            float(trapezoid(dG_nz.cpu().numpy(), bin_centers_nz.cpu().numpy()))
        )

        mask = torch.abs(dG_nz) > 1e-4 * torch.abs(integral)
        n_degs = len(self.theta_names)
        try:
            # Determine best fit for unknown # of parameters 
            p0s = lambda m: [3.14*(1-m+2*i)/(2*m) for i in range(m)]
            popts = []
            aics = []
            for i_cd in range(n_degs):
                p0 = []
                for ip in range(n_degs):
                    if ip > i_cd:
                        p0.append(0)
                        p0.append(0)
                    else:
                        p0.append(p0s[i_cd][ip])
                        p0.append(1)
                dihedral_compute = Dihedral().ComputeClass()
                dihedral_compute.deg = i_cd
                free_parameters = 2*(i_cd+1)

                popt, _ = curve_fit(
                    dihedral_compute.compute,
                    bin_centers_nz[mask],
                    dG_nz[mask],
                    p0,
                )
                popts.append(popt)
                aic = (
                    2
                    * Dihedral.neg_log_likelihood(
                        dG_nz[mask],
                        dihedral_compute.compute(bin_centers_nz[mask], *popt),
                    )
                    - 2 * free_parameters
                )
                aics.append(aic)

            min_aic = min(aics)
            min_i_aic = aics.index(min_aic)

            popt = popts[min_i_aic]
            for ii in range(n_degs):
                stat["theta"][ii] = popt[2*ii]
                stat["k"][ii] = popt[2*ii+1] 
            dihedral_compute.deg = min_i_aic
            Dihedral.compute = dihedral_compute.compute 

        except:
            print(f"failed to fit potential estimate for Dihedral")
            for ii in range(n_degs):
                stat["theta"][ii] = torch.tensor(float("nan"))
                stat["k"][ii] = torch.tensor(float("nan"))
        return stat

    def from_user(*args):
        """
        Direct input of parameters from user. Leave empty for now
        """
        stat = {}
        # stat = {
        #     "theta_0": args[0],
        #     "k_0": args[1],
        #     "theta_1": args[2],
        #     "k_1": args[3],
        #     "theta_2": args[4],
        #     "k_2": args[5],
        # }
        return stat

    @staticmethod
    def neighbor_list(topology) -> None:
        nl = topology.neighbor_list("dihedrals")
        return {Dihedral._name: nl}
