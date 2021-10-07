
# Workaround for apple M1 which does not support mdtraj in a simple manner
try:
    import mdtraj
except ModuleNotFoundError:
    print(f'Failed to import mdtraj')

from typing import Tuple, Dict, NamedTuple, List, Optional
import numpy as np

from ._mappings import CA_MAP

class Atom(NamedTuple):
    """Define an atom
    """
    #: type of the atom
    type: int
    #: name of the atom
    name: Optional[str] = None
    #: name of the residue containing the atom
    resname: Optional[str] = None


class Topology(NamedTuple):
    """Define the topology of an isolated protein.
    """
    #: types of the atoms
    types: List[int] = []
    #: name of the atoms
    names: List[str] = []
    #: name of the residue containing the atoms
    resnames: List[str] = []
    #: list of bonds between the atoms
    bonds: List[List[int,int]] = []
    #: list of angles formed by triplets of atoms
    angles: List[List[int,int,int]] = []
    #: list of dihedrals formed by quadruplets of atoms
    dihedrals: List[List[int,int,int,int]] = []

    def add_atom(self, type:int, name:str, resname:str):
        self.types.append(type)
        self.names.append(name)
        self.resnames.append(resname)

    def atoms(self):
        for type,name,resname in zip(self.types, self.names, self.resnames):
            yield Atom(type=type,name=name,resname=resname)

    @property
    def n_atoms(self)->int:
        """Number of atoms in the topology.
        """
        return self.types.shape[0]

    def add_bond(self, idx1:int, idx2:int):
        """Define a bond between two atoms.

        Parameters
        ----------
        idx:
            index of the atoms bonded together
        """
        self.bonds.append([idx1, idx2])

    def add_angle(self, idx1:int, idx2:int, idx3:int):
        """Define an angle between three atoms. `idx2` represent the apex of
        the angles::

          2---3
         /
        1
        """
        self.angles.append([idx1, idx2, idx3])

    def add_dihedral(self, idx1:int, idx2:int, idx3:int, idx4:int):
        """
        The dihedral angle formed by a quadruplet of indices (1,2,3,4) is
        difined around the axis connecting index 2 and 3 (i.e., the angle
        between the planes spanned by indices (1,2,3) and (2,3,4))::

                  4
                  |
            2-----3
           /
          1
        """
        self.dihedrals.append([idx1, idx2, idx3, idx4])

    def to_mdtraj(self):
        """Convert to mdtraj format
        """
        topo = mdtraj.Topology()
        chain = topo.add_chain()
        for i_at in range(self.n_atoms):
            residue = topo.add_residue(self.resnames[i_at],chain)
            topo.add_atom(self.names[i_at],self.types[i_at],residue)
        for idx1, idx2 in self.bonds:
            a1,a2 = topo.atom(idx1), topo.atom(idx2)
            topo.add_bond(a1,a2)
        return topo

    @staticmethod
    def from_mdtraj(topology):
        """Build topology from an existing mdtraj topology.
        """
        assert topology.n_chains == 1, f"Does not support multiple chains but {topology.n_chains}"
        topo = Topology()
        for at in topology.atoms:
            topo.add_atom(at.element, at.name, at.residue.name)
        for at1,at2 in topology.bonds:
            topo.add_bond(at1.index, at2.index)
        return topo

    @staticmethod
    def from_file(filename:str):
        """Uses mdtraj reader to read the input topology.
        """
        topo = mdtraj.load(filename).topology
        return Topology.from_mdtraj(topo)


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
        if ((i_at == 0) or (i_at == n_atoms-1)) and special_terminal:
            cg_name += '-terminal'
        cg_topo.add_atom(cg_type, cg_name, at.resname)

    if bonds:
        for i in range(cg_topo.n_atoms-1):
            cg_topo.add_bond(i, i+1)
    if angles:
        for i in range(cg_topo.n_atoms-2):
            cg_topo.add_angle(i, i+1, i+2)

    return cg_topo