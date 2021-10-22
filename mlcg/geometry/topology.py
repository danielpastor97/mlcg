# Workaround for apple M1 which does not support mdtraj in a simple manner
import warnings

try:
    import mdtraj
except ModuleNotFoundError:
    warnings(f"Failed to import mdtraj")

from typing import NamedTuple, List, Optional, Tuple
import torch

from ..neighbor_list.neighbor_list import make_neighbor_list


class Atom(NamedTuple):
    """Define an atom"""

    #: type of the atom
    type: int
    #: name of the atom
    name: Optional[str] = None
    #: name of the residue containing the atom
    resname: Optional[str] = None


class Topology(object):
    """Define the topology of an isolated protein."""

    #: types of the atoms
    types: List[int]
    #: name of the atoms
    names: List[str]
    #: name of the residue containing the atoms
    resnames: List[str]
    #: list of bonds between the atoms
    bonds: Tuple[List[int], List[int]]
    #: list of angles formed by triplets of atoms
    angles: Tuple[List[int], List[int], List[int]]
    #: list of dihedrals formed by quadruplets of atoms
    dihedrals: Tuple[List[int], List[int], List[int], List[int]]

    def __init__(self) -> None:
        super(Topology, self).__init__()
        self.types = []
        self.names = []
        self.resnames = []
        self.bonds = ([], [])
        self.angles = ([], [], [])
        self.dihedrals = ([], [], [], [])

    def add_atom(self, type: int, name: str, resname: Optional[str] = None):
        self.types.append(type)
        self.names.append(name)
        self.resnames.append(resname)

    @property
    def atoms(self):
        for type, name, resname in zip(self.types, self.names, self.resnames):
            yield Atom(type=type, name=name, resname=resname)

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the topology."""
        return len(self.types)

    def types2torch(self, device: str = "cpu"):
        return torch.tensor(self.types, dtype=torch.long, device=device)

    def bonds2torch(self, device: str = "cpu"):
        return torch.tensor(self.bonds, dtype=torch.long, device=device)

    def angles2torch(self, device: str = "cpu"):
        return torch.tensor(self.angles, dtype=torch.long, device=device)

    def dihedrals2torch(self, device: str = "cpu"):
        return torch.tensor(self.dihedrals, dtype=torch.long, device=device)

    def fully_connected2torch(self, device: str = "cpu"):
        ids = torch.arange(self.n_atoms)
        mapping = torch.cartesian_prod(ids, ids).t()
        mapping = mapping[:, mapping[0] != mapping[1]]
        return mapping

    def neighbor_list(self, type: str, device: str = "cpu"):
        """Build Neighborlist from a :ref:`mlcg.neighbor_list.neighbor_list.Topology`.

        Parameters
        ----------
        type:
            kind of information to extract (should be in ["bonds", "angles",
            "dihedrals", "fully connected"]).
        device:
            device onto which to return the nl
        """
        allowed_types = ["bonds", "angles", "dihedrals", "fully connected"]
        assert type in allowed_types, f"type should be any of {allowed_types}"
        if type == "bonds":
            mapping = self.bonds2torch(device)
        elif type == "angles":
            mapping = self.angles2torch(device)
        elif type == "dihedrals":
            mapping = self.dihedrals2torch(device)
        elif type == "fully connected":
            mapping = self.fully_connected2torch(device)

        nl = make_neighbor_list(
            tag=type,
            order=mapping.shape[0],
            index_mapping=mapping,
            self_interaction=False,
        )
        return nl

    def add_bond(self, idx1: int, idx2: int):
        """Define a bond between two atoms.

        Parameters
        ----------
        idx:
            index of the atoms bonded together
        """
        self.bonds[0].append(idx1)
        self.bonds[1].append(idx2)

    def add_angle(self, idx1: int, idx2: int, idx3: int):
        """Define an angle between three atoms. `idx2` represent the apex of
        the angles::

          2---3
         /
        1
        """
        self.angles[0].append(idx1)
        self.angles[1].append(idx2)
        self.angles[2].append(idx3)

    def add_dihedral(self, idx1: int, idx2: int, idx3: int, idx4: int):
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
        self.dihedrals[0].append(idx1)
        self.dihedrals[1].append(idx2)
        self.dihedrals[2].append(idx3)
        self.dihedrals[3].append(idx4)

    def to_mdtraj(self):
        """Convert to mdtraj format"""
        topo = mdtraj.Topology()
        chain = topo.add_chain()
        for i_at in range(self.n_atoms):
            residue = topo.add_residue(self.resnames[i_at], chain)
            topo.add_atom(self.names[i_at], self.types[i_at], residue)
        for idx1, idx2 in self.bonds:
            a1, a2 = topo.atom(idx1), topo.atom(idx2)
            topo.add_bond(a1, a2)
        return topo

    @staticmethod
    def from_mdtraj(topology):
        """Build topology from an existing mdtraj topology."""
        assert (
            topology.n_chains == 1
        ), f"Does not support multiple chains but {topology.n_chains}"
        topo = Topology()
        for at in topology.atoms:
            topo.add_atom(at.element.atomic_number, at.name, at.residue.name)
        for at1, at2 in topology.bonds:
            topo.add_bond(at1.index, at2.index)
        return topo

    @staticmethod
    def from_file(filename: str):
        """Uses mdtraj reader to read the input topology."""
        topo = mdtraj.load(filename).topology
        return Topology.from_mdtraj(topo)


def add_chain_bonds(topology: Topology) -> None:
    """Add bonds to the topology assuming a chain-like pattern, i.e. atoms are
    linked together following their insertion order.
    A four atoms chain will are linked like: `1-2-3-4`.
    """
    for i in range(topology.n_atoms - 1):
        topology.add_bond(i, i + 1)


def add_chain_angles(topology: Topology) -> None:
    """Add angles to the topology assuming a chain-like pattern, i.e. angles are
    defined following the insertion order of the atoms in the topology.
    A four atoms chain `1-2-3-4` will fine the angles: `1-2-3, 2-3-4`.
    """
    for i in range(topology.n_atoms - 2):
        topology.add_angle(i, i + 1, i + 2)

def add_chain_dihedrals(topology: Topology) -> None:
    """Add dihedrals to the topology assuming a chain-like pattern, i.e. dihedrals are
    defined following the insertion order of the atoms in the topology.
    A four atoms chain `1-2-3-4` will find the dihedral: `1-2-3-4`.
    """
    for i in range(topology.n_atoms - 3):
        topology.add_dihedral(i, i + 1, i + 2, i+3)       
