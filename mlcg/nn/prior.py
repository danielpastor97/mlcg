import torch
from torch_scatter import scatter
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
from typing import Final, Optional
from math import pi

from ..geometry.topology import Topology
from ..geometry.internal_coordinates import (
    compute_distances,
    compute_angles,
    compute_dihedrals,
    compute_impropers,
)


class _Prior(object):
    def __init__(self) -> None:
        super(_Prior, self).__init__()


class Harmonic(torch.nn.Module, _Prior):
    _order_map = {
        "bonds": 2,
        "angles": 3,
        "impropers": 4,
    }
    _compute_map = {
        "bonds": compute_distances,
        "angles": compute_angles,
        "impropers": compute_impropers,
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


class HarmonicImpropers(Harmonic):
    name: Final[str] = "impropers"
    _order = 2

    def __init__(self, statistics) -> None:
        super(HarmonicImpropers, self).__init__(
            statistics, HarmonicImpropers.name
        )

    @staticmethod
    def neighbor_list(topology: Topology) -> dict:
        return Harmonic.neighbor_list(topology, HarmonicImpropers.name)

    @staticmethod
    def compute_features(pos, mapping):
        return Harmonic.compute_features(pos, mapping, HarmonicImpropers.name)


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

    def __init__(self, statistics, n_degs=6) -> None:
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
        self.n_degs = n_degs
        self.k1_names = ["k1_" + str(ii) for ii in range(1, self.n_degs)]
        self.k2_names = ["k2_" + str(ii) for ii in range(1, self.n_degs)]
        k1 = self.n_degs * [torch.zeros(sizes)]
        k2 = self.n_degs * [torch.zeros(sizes)]

        for key in statistics.keys():
            for ii in range(self.n_degs):
                k1_name = self.k1_names[ii]
                k2_name = self.k2_names[ii]
                k1[ii][key] = statistics[key]["k1s"][k1_name]
                k2[ii][key] = statistics[key]["k2s"][k2_name]
        for ii in range(self.n_degs):
            k1_name = self.k1_names[ii]
            k2_name = self.k2_names[ii]
            self.register_buffer(k1_name, k1[ii])
            self.register_buffer(k2_name, k2[ii])

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
        k1s = []
        k2s = []
        for ii in range(self.n_degs):
            k1_name = self.k1_names[ii]
            k2_name = self.k2_names[ii]
            k1s.append(getattr(self, k1_name)[interaction_types])
            k2s.append(getattr(self, k2_name)[interaction_types])
        y = Dihedral.compute(
            features,
            k1s,
            k2s,
        )
        y = scatter(y, mapping_batch, dim=0, reduce="sum")
        data.out[self.name] = {"energy": y}
        return data

    @staticmethod
    def compute_features(pos, mapping):
        return compute_dihedrals(pos, mapping)

    @staticmethod
    def wrapper_fit_func(theta, *args):
        n_k1s = int(len(args[0]) / 2)
        k1s, k2s = list(args[0][:n_k1s]), list(args[0][n_k1s:])
        k1s = torch.tensor(k1s)
        k2s = torch.tensor(k2s)
        return Dihedral.compute(theta, k1s, k2s)

    @staticmethod
    def compute(theta, k1s, k2s):
        V = 0.00
        for ii, (k1, k2) in enumerate(zip(k1s, k2s)):
            V += k1 * torch.sin(ii * theta) + k2 * torch.cos(ii * theta)
        return V

    @staticmethod
    def neg_log_likelihood(y, yhat):
        """
        Convert dG to probability and use KL divergence to get difference between
        predicted and actual
        """
        L = torch.sum(torch.exp(-y) * torch.log(torch.exp(-yhat)))
        return -L

    @staticmethod
    def _init_parameters(n_degs):
        """Helper method for guessing initial parameter values"""
        p0 = []
        k1s_0 = [1 for _ in range(n_degs)]
        k2s_0 = [1 for _ in range(n_degs)]
        p0.append(k1s_0)
        p0.append(k2s_0)
        return p0

    @staticmethod
    def _init_parameter_dict(n_degs):
        """Helper method for initializing the parameter dictionary"""
        stat = {
            "k1s": {},
            "k2s": {},
        }
        k1_names = ["k1_" + str(ii) for ii in range(n_degs)]
        k2_names = ["k2_" + str(ii) for ii in range(n_degs)]
        for ii in range(n_degs):
            k1_name = k1_names[ii]
            k2_name = k2_names[ii]
            stat["k1s"][k1_name] = {}
            stat["k2s"][k2_name] = {}
        return stat

    @staticmethod
    def _make_parameter_dict(stat, popt, n_degs):
        """Helper method for constructing a fitted parameter dictionary"""
        num_k1s = int(len(popt) / 2)
        k1_names = sorted(list(stat["k1s"].keys()))
        k2_names = sorted(list(stat["k2s"].keys()))
        for ii in range(n_degs):
            k1_name = k1_names[ii]
            k2_name = k2_names[ii]
            stat["k1s"][k1_name] = {}
            stat["k2s"][k2_name] = {}
            if len(popt) > 2 * ii:
                stat["k1s"][k1_name] = popt[ii]
                stat["k2s"][k2_name] = popt[num_k1s + ii]
            else:
                stat["k1s"][k1_name] = 0
                stat["k2s"][k2_name] = 0
        return stat

    @staticmethod
    def _compute_aic(bin_centers_nz, dG_nz, mask, popt, free_parameters):
        """Method for computing the AIC"""
        aic = (
            2
            * Dihedral.neg_log_likelihood(
                dG_nz[mask],
                Dihedral.wrapper_fit_func(bin_centers_nz[mask], *[popt]),
            )
            + 2 * free_parameters
        )
        return aic

    @staticmethod
    def fit_from_potential_estimates(
        bin_centers_nz: torch.Tensor,
        dG_nz: torch.Tensor,
        n_degs: int = 6,
        constrain_deg: Optional[int] = None,
    ):
        """
        Loop over n_degs basins and use either the AIC criterion
        or a prechosen degree to select best fit. Parameter fitting
        occurs over unmaksed regions of the free energy only.

        Parameters
        ----------
        bin_centers_nz:
            Bin centers over which the fit is carried out
        dG_nz:
            The emperical free energy correspinding to the bin centers
        n_degs:
            The maximum number of degrees to attempt to fit if using the AIC
            criterion for prior model selection
        constrain_deg:
            If not None, a single fit is produced for the specified integer
            degree instead of using the AIC criterion for fit selection between
            multiple degrees

        Returns
        -------
            Statistics dictionary with fitted interaction parameters
        """

        integral = torch.tensor(
            float(trapezoid(dG_nz.cpu().numpy(), bin_centers_nz.cpu().numpy()))
        )

        mask = torch.abs(dG_nz) > 1e-4 * torch.abs(integral)

        if constrain_deg != None:
            assert isinstance(constrain_deg, int)
            stat = Dihedral._init_parameter_dict(constrain_deg)
            p0 = Dihedral._init_parameters(constrain_deg)
            popt, _ = curve_fit(
                lambda theta, *p0: Dihedral.wrapper_fit_func(theta, p0),
                bin_centers_nz[mask],
                dG_nz[mask],
                p0=p0,
            )
            stat = Dihedral._make_parameter_dict(
                stat, popt, constrain_deg
            )

        else:
            try:
                # Determine best fit for unknown # of parameters
                stat = Dihedral._init_parameter_dict(n_degs)
                popts = []
                aics = []

                for deg in range(1, n_degs + 1):
                    p0 = Dihedral._init_parameters(deg)
                    free_parameters = 2 * (deg + 1)

                    popt, _ = curve_fit(
                        lambda theta, *p0: Dihedral.wrapper_fit_func(theta, p0),
                        bin_centers_nz[mask],
                        dG_nz[mask],
                        p0=p0,
                    )
                    aic = Dihedral._compute_aic(
                        bin_centers_nz, dG_nz, mask, popt, free_parameters
                    )
                    popts.append(popt)
                    aics.append(aic)
                min_aic = min(aics)
                min_i_aic = aics.index(min_aic)
                popt = popts[min_i_aic]
                stat = Dihedral._make_parameter_dict(stat, popt, n_degs)
            except:
                print(f"failed to fit potential estimate for Dihedral")
                stat = Dihedral._init_parameter_dict(n_degs)
                k1_names = sorted(list(stat["k1s"].keys()))
                k2_names = sorted(list(stat["k2s"].keys()))
                for ii in range(n_degs):
                    k1_name = k1_names[ii]
                    k2_name = k2_names[ii]
                    stat["k1s"][k1_name] = torch.tensor(float("nan"))
                    stat["k2s"][k2_name] = torch.tensor(float("nan"))
        return stat

    def from_user(*args):
        """
        Direct input of parameters from user. Leave empty for now
        """
        raise NotImplementedError()

    @staticmethod
    def neighbor_list(topology) -> None:
        nl = topology.neighbor_list("dihedrals")
        return {Dihedral._name: nl}
