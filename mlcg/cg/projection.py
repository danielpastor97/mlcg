from typing import Optional, Tuple, Dict, Callable, OrderedDict
from collections import OrderedDict
import numpy as np


from ._mappings import CA_MAP
from ..geometry.topology import (
    Topology,
    add_chain_bonds,
    add_chain_angles,
    add_chain_dihedrals,
)


def build_cg_matrix(
    topology: Topology,
    cg_mapping: Dict[Tuple[str, str], Tuple[str, int, int]] = None,
    special_terminal: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, OrderedDict]:
    r"""Function for producing coarse grain types, masses, and
    mapping matrices using a slicing strategy and a predetermined
    set of atoms to retain at the coarse grain resolution.

    Parameters
    ----------
    topology:
        System topology instance
    cg_mapping:
        Mapping dictionary with the following structure:

        .. code_block:: python

            {
                (residue name, atom name) : (compound name, type, mass)
                ...
            }

        Eg, a row for an alanine carbon alpha atom would be:

        .. code-block:: python

            {
                ("ALA", "CA") : ("CA_A", 1, 12)
                ...
            }

    special_termini:
        If True, special types will be reserved for the first and
        last CG atoms

    Returns
    -------
    cg_types:
        Array of CG atom types
    cg_masses:
        Array of CG masses
    cg_matrix:
        One-hot transformation matrix of shape (n_high_res_atoms, n_cg_atoms)
        that maps atoms for the high resolution repesentation to the coarse
        grain representation
    cg_mapping_:
        Ordered dictionary mapping each CG atom index (with respect to the
        CG topology) to a list containing the CG atom name, CG atom type
        and the CG atom mass
    """

    if cg_mapping == None:
        cg_mapping = CA_MAP

    cg_mapping_ = OrderedDict()
    n_atoms = topology.n_atoms
    for i_at, at in enumerate(topology.atoms):
        (cg_name, cg_type, cg_mass) = cg_mapping.get(
            (at.resname, at.name), (None, None, None)
        )
        if cg_name is None:
            continue

        cg_mapping_[i_at] = [cg_name, cg_type, cg_mass]

    if special_terminal:
        keys = list(cg_mapping_)
        cg_mapping_[keys[0]][0] += "-terminal"
        cg_mapping_[keys[0]][1] += len(cg_mapping)
        cg_mapping_[keys[-1]][0] += "-terminal"
        cg_mapping_[keys[-1]][1] += len(cg_mapping)

    n_beads = len(cg_mapping_)

    cg_types = np.array([cg_type for (_, cg_type, _) in cg_mapping_.values()])
    cg_masses = np.array([cg_mass for (_, _, cg_mass) in cg_mapping_.values()])

    cg_matrix = np.zeros((n_beads, n_atoms))
    for i_cg, i_at in enumerate(cg_mapping_.keys()):
        cg_matrix[i_cg, i_at] = 1

    return cg_types, cg_masses, cg_matrix, cg_mapping_


def build_cg_topology(
    topology: Topology,
    cg_mapping: Dict[Tuple[str, str], Tuple[str, int, int]] = None,
    special_terminal: bool = True,
    bonds: Optional[Callable] = add_chain_bonds,
    angles: Optional[Callable] = add_chain_angles,
    dihedrals: Optional[Callable] = add_chain_dihedrals,
):
    """Function for building a coarse grain topology
    from a high resolution topology and a coarse grain mapping

    Parameters
    ----------
    topology:
        Topology of the high resolution system
    cg_mapping:
        Ordered dictionary mapping each CG atom index (with respect to the
        CG topology) to a list containing the CG atom name, CG atom type
        and the CG atom mass
    special_termini:
        If True, special types will be reserved for the first and
        last CG atoms
    bonds:
        Function that assigns bonds to the coarse grain topology
    angles:
        Function that assigns angles to the coarse grain topology

    Returns
    -------
    cg_topo:
        Coarse grain topology
    """
    if cg_mapping == None:
        cg_mapping = CA_MAP

    cg_topo = Topology()
    for at in topology.atoms:
        (cg_name, cg_type, _) = cg_mapping.get(
            (at.resname, at.name), (None, None, None)
        )
        if cg_name is None:
            continue
        cg_topo.add_atom(cg_type, cg_name, at.resname)

    if special_terminal:
        cg_topo.names[0] += "-terminal"
        cg_topo.types[0] += len(cg_mapping)
        cg_topo.names[-1] += "-terminal"
        cg_topo.types[-1] += len(cg_mapping)

    if bonds is not None:
        bonds(cg_topo)
    if angles is not None:
        angles(cg_topo)
    if dihedrals is not None:
        dihedrals(cg_topo)

    return cg_topo
