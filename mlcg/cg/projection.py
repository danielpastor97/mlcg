from typing import Tuple, Dict
from collections import OrderedDict
import numpy as np


from ._mappings import CA_MAP
from ..geometry.topology import Topology


def build_cg_matrix(
    topology,
    cg_mapping: Dict[Tuple[str, str], Tuple[str, int]] = CA_MAP,
    special_terminal: bool = True,
):
    cg_mapping_ = OrderedDict()
    n_atoms = topology.n_atoms
    for i_at, at in enumerate(topology.atoms):
        (cg_name, cg_type) = cg_mapping.get(
            (at.residue.name, at.name), (None, None)
        )

        if cg_name is None:
            continue
        if ((i_at == 0) or (i_at == n_atoms - 1)) and special_terminal:
            cg_name += "-terminal"
            cg_type += len(cg_mapping)

        cg_mapping_[i_at] = (cg_name, cg_type)

    n_beads = 2 * len(cg_mapping)

    cg_types = np.array([cg_type for (_, cg_type) in cg_mapping_.values()])

    cg_matrix = np.zeros((n_beads, n_atoms))

    for i_at, (cg_name, cg_type) in cg_mapping_.items():
        cg_matrix[cg_type, i_at] = 1

    return cg_types, cg_matrix, cg_mapping_


def build_cg_topology(
    topology,
    cg_mapping: Dict[Tuple[str, str], Tuple[str, int]] = CA_MAP,
    special_terminal: bool = True,
    bonds: bool = True,
    angles: bool = True,
):
    cg_topo = Topology()
    n_atoms = topology.n_atoms
    for i_at, at in enumerate(topology.atoms()):
        (cg_name, cg_type) = cg_mapping.get((at.resname, at.name), (None, None))

        if cg_name is None:
            continue
        if ((i_at == 0) or (i_at == n_atoms - 1)) and special_terminal:
            cg_name += "-terminal"
        cg_topo.add_atom(cg_type, cg_name, at.resname)

    if bonds:
        for i in range(cg_topo.n_atoms - 1):
            cg_topo.add_bond(i, i + 1)
    if angles:
        for i in range(cg_topo.n_atoms - 2):
            cg_topo.add_angle(i, i + 1, i + 2)

    return cg_topo
