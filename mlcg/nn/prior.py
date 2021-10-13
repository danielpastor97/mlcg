import torch
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit

from mlcg.data import atomic_data

from ..geometry.internal_coordinates import compute_distances ,compute_angles
from ..neighbor_list.neighbor_list import topology2neighbor_list, atomic_data2neighbor_list


def tensor2tuple(t):
    return tuple([v for v in t.flatten()])

class _Prior(object):
    def __init__(self) -> None:
        super(_Prior, self).__init__()

class Harmonic(torch.nn.Module, _Prior):

    _neighbor_list_names = {2:'bonds', 3:'angles'}
    _compute_map = {
        2: compute_distances,
        3: compute_angles,
    }

    def __init__(self, statistics):
        super(Harmonic, self).__init__()
        keys = torch.vstack(list(statistics.keys()))
        self.allowed_interaction_keys = keys
        self.order = keys.shape[1]
        self.neighbor_list_name = self._neighbor_list_names[self.order]
        unique_types = torch.unique(keys.flatten())
        assert unique_types.min() < 0
        max_type = unique_types.max()
        sizes = tuple([max_type+1 for _ in range(self.order)])
        x_0 = torch.zeros(sizes)
        k = torch.zeros(sizes)
        for k in keys:
            x_0[k.view(self.order, 1)] = statistics[k]['x_0']
            k[k.view(self.order, 1)] = statistics[k]['k']

        self.register_buffer('x_0', x_0)
        self.register_buffer('k', k)

    def data2features(self, data):
        mapping = data.neighbor_list[self.neighbor_list_name]['index_mapping']
        return Harmonic.compute_features(data.pos, mapping, self.order)

    def forward(self, data):
        mapping = data.neighbor_list[self.neighbor_list_name]['index_mapping']
        interaction_types = [data.atom_types[mapping[ii]] for ii in range(self.order)]
        features = self.data2features(data).flatten()
        return Harmonic.eval(features, self.x_0[interaction_types], self.k[interaction_types], 0)

    @staticmethod
    def compute_features(pos, mapping, order, **kwargs):
        return Harmonic._compute_map[order](pos, mapping)

    @staticmethod
    def eval(x,x0,k,V0):
        return k*(x-x0)**2+V0

    @staticmethod
    def fit_from_data(bin_centers_nz, dG_nz):
        # remove noise by discarding signals
        integral = trapezoid(dG_nz.cpu().numpy(), bin_centers_nz.cpu().numpy())
        print(integral)
        mask = torch.abs(dG_nz) > 1e-4*torch.abs(integral)
        try:
            popt, _ = curve_fit(Harmonic.eval, bin_centers_nz[mask], dG_nz[mask],
                        p0=[bin_centers_nz[torch.argmin(dG_nz[mask])[0]], 60, -1])
            stat = {'k':popt[1],'x_0':popt[0]}
        except:
            stat = {'k':torch.nan,'x_0':torch.nan}
        return stat

    @staticmethod
    def neighbor_list(topology, type) -> None:
        assert type in Harmonic._neighbor_list_names
        nl = topology2neighbor_list(topology, type=type)
        return {type: nl}

class Repulsion(torch.nn.Module, _Prior):

    _neighbor_list_name = 'repulsion'

    def __init__(self, statistics):
        super(Repulsion, self).__init__()
        keys = torch.vstack(list(statistics.keys()))
        self.allowed_interaction_keys = keys
        self.order = 2
        self.neighbor_list_name = self._neighbor_list_name
        unique_types = torch.unique(keys.flatten())
        assert unique_types.min() < 0
        max_type = unique_types.max()
        sizes = tuple([max_type+1 for _ in range(self.order)])
        sigma = torch.zeros(sizes)
        for k in keys:
            sigma[k.view(self.order, 1)] = statistics[k]['sigma']

        self.register_buffer('sigma', sigma)

    def data2features(self, data):
        mapping = data.neighbor_list[self.neighbor_list_name]['index_mapping']
        return Repulsion.compute_features(data.pos, mapping)

    def forward(self, data):
        mapping = data.neighbor_list[self.neighbor_list_name]['index_mapping']
        interaction_types = [data.atom_types[mapping[ii]] for ii in range(self.order)]
        features = self.data2features(data)
        return Repulsion.eval(features, self.sigma[interaction_types])

    @staticmethod
    def compute_features(pos, mapping, **kwargs):
        return compute_distances(pos, mapping)

    @staticmethod
    def eval(x,sigma):
        rr = (sigma/x) * (sigma/x)
        return rr*rr*rr

    @staticmethod
    def fit_from_data(bin_centers_nz, dG_nz):
        delta = bin_centers_nz[1] - bin_centers_nz[0]
        sigma = bin_centers_nz[0] - 0.5*delta
        stat = {'sigma':sigma}
        return stat

    @staticmethod
    def neighbor_list(data, rcut=5) -> None:
        nl = atomic_data2neighbor_list(data, rcut=rcut, self_interaction=False)
        return {Repulsion._neighbor_list_name: nl}