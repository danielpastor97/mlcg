
# Workaround for apple M1 which does not support mdtraj in a simple manner
try:
    import mdtraj
except ModuleNotFoundError:
    print(f'Failed to import mdtraj')


from typing import Tuple, Dict
from collections import OrderedDict
import numpy as np


from ._mappings import CA_MAP

def build_cg_topology(
    topology,
    cg_mapping: Dict[Tuple[str, str], Tuple[str, int]] = CA_MAP,
    special_terminal: bool = True,
):
    cg_topo = mdtraj.Topology()
    chain = cg_topo.add_chain()
    n_atoms = topology.n_atoms
    for i_at, at in enumerate(topology.atoms):
        (cg_name, _) = cg_mapping.get((at.residue.name, at.name), (None, None))

        if cg_name is None:
            continue
        if ((i_at == 0) or (i_at == n_atoms-1)) and special_terminal:
            cg_name += '-terminal'

        residue = cg_topo.add_residue(at.residue.name,chain)
        cg_topo.add_atom(at.name,at.element,residue)

    for i in range(cg_topo.n_atoms-1):
        a1,a2 = cg_topo.atom(i), cg_topo.atom(i+1)
        cg_topo.add_bond(a1,a2)
    return cg_topo