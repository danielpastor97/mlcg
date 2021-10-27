import torch
from torch_geometric.data import Data
from ase.geometry.analysis import Analysis

def ase2data(frame, energy_tag=None, force_tag=None):
    z = torch.from_numpy(frame.get_atomic_numbers())
    pos = torch.from_numpy(frame.get_positions())
    pbc = torch.from_numpy(frame.get_pbc())
    cell = torch.tensor(frame.get_cell().tolist(), dtype=torch.float64)
    n_atoms = torch.tensor([len(frame)])
    data = Data(z=z, pos=pos, pbc=pbc, cell=cell, n_atoms=n_atoms)

    if energy_tag is not None:
        E = torch.tensor(frame.info[energy_tag])
        data.energy = E
    if force_tag is not None:
        forces = torch.from_numpy(frame.arrays[force_tag])
        data.forces = forces

    return data


def ase_bonds2tensor(analysis: Analysis, unique=True) -> torch.Tensor:
    """converts ASE single neighborlist bond list to tensor of shape (2, n_bonds)"""
    if unique:
        bond_list = analysis.unique_bonds
    else:
        bond_list = analysis.all_bonds
    edge_tensor = torch.tensor([[],[]])
    for atom, neighbors in enumerate(bond_list[0]):
        for bonded_neighbor in neighbors:
            edge_tensor = torch.cat((edge_tensor, torch.tensor([[atom],[bonded_neighbor]])), dim=1)
    return edge_tensor.long()


def ase_angles2tensor(analysis: Analysis, unique=True) -> torch.Tensor:
    """converts ASE single neighborlist angle list to tensor of shape (2, n_bonds)"""
    if unique:
        angle_list = analysis.unique_angles
    else:
        angle_list = analysis.all_angles
    edge_tensor = torch.tensor([[],[],[]])
    for atom, end_point_list in enumerate(angle_list[0]):
        for end_points in end_point_list:
            edge_tensor = torch.cat((edge_tensor, torch.tensor([[end_points[0]],[atom],[end_points[1]]])), dim=1)
    return edge_tensor.long()
