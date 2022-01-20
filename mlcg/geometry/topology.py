# Workaround for apple M1 which does not support mdtraj in a simple manner
import warnings

try:
    import mdtraj
    from mdtraj.core.element import Element
except ModuleNotFoundError:
    warnings(f"Failed to import mdtraj")
from ase.geometry.analysis import Analysis
from ase import Atoms
from typing import NamedTuple, List, Optional, Tuple, Dict, Callable
import torch
import numpy as np
import networkx as nx
from itertools import combinations

from .utils import ase_z2name
from ..neighbor_list.neighbor_list import make_neighbor_list
from ._symmetrize import (
    _symmetrise_map,
    _symmetrise_angle_interaction,
    _symmetrise_distance_interaction,
)


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
    #: list of bonds between the atoms. Defines the bonded topology.
    bonds: Tuple[List[int], List[int]]
    #: list of angles formed by triplets of atoms
    angles: Tuple[List[int], List[int], List[int]]
    #: list of dihedrals formed by quadruplets of atoms
    dihedrals: Tuple[List[int], List[int], List[int], List[int]]
    #: list of impropers formed by quadruplets of atoms
    impropers: Tuple[List[int], List[int], List[int], List[int]]

    def __init__(self) -> None:
        super(Topology, self).__init__()
        self.types = []
        self.names = []
        self.resnames = []
        self.bonds = ([], [])
        self.angles = ([], [], [])
        self.dihedrals = ([], [], [], [])
        self.impropers = ([], [], [], [])

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

    def impropers2torch(self, device: str = "cpu"):
        return torch.tensor(self.impropers, dtype=torch.long, device=device)

    def fully_connected2torch(self, device: str = "cpu"):
        ids = torch.arange(self.n_atoms)
        mapping = torch.cartesian_prod(ids, ids).t()
        mapping = mapping[:, mapping[0] != mapping[1]]
        return mapping

    def neighbor_list(self, type: str, device: str = "cpu") -> Dict:
        """Build Neighborlist from a :ref:`mlcg.neighbor_list.neighbor_list.Topology`.

        Parameters
        ----------
        type:
            kind of information to extract (should be in ["bonds", "angles",
            "dihedrals", "fully connected"]).
        device:
            device upon which the neighborlist is returned
        """
        allowed_types = [
            "bonds",
            "angles",
            "dihedrals",
            "impropers",
            "fully connected",
        ]
        assert type in allowed_types, f"type should be any of {allowed_types}"
        if type == "bonds":
            mapping = self.bonds2torch(device)
        elif type == "angles":
            mapping = self.angles2torch(device)
        elif type == "dihedrals":
            mapping = self.dihedrals2torch(device)
        elif type == "impropers":
            mapping = self.impropers2torch(device)
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

    def bonds_from_edge_index(self, edge_index: torch.tensor):
        """Overwrites the internal bond list with the bonds
        defined in the supplied bond edge_index

        Parameters
        ----------
        edge_index:
            Edge index tensor of shape (2, n_bonds)
        """
        if edge_index.shape[0] != 2:
            raise ValueError("Bond edge index must have shape (2, n_bonds)")

        self.bonds = tuple(edge_index.numpy().tolist())

    def angles_from_edge_index(self, edge_index: torch.tensor):
        """Overwrites the internal angle list with the angles
        defined in the supplied angle edge_index

        Parameters
        ----------
        edge_index:
            Edge index tensor of shape (3, n_angles)
        """
        if edge_index.shape[0] != 3:
            raise ValueError("Angle edge index must have shape (3, n_angles)")

        self.angles = tuple(edge_index.numpy().tolist())

    def dihedrals_from_edge_index(self, edge_index: torch.tensor):
        """Overwrites the internal dihedral list with the dihedral
        defined in the supplied dihedral edge_index

        Parameters
        ----------
        edge_index:
            Edge index tensor of shape (4, n_dihedrals)
        """
        if edge_index.shape[0] != 4:
            raise ValueError(
                "Dihedral edge index must have shape (4, n_dihedrals)"
            )

        self.dihedrals = tuple(edge_index.numpy().tolist())

    def impropers_from_edge_index(self, edge_index: torch.tensor):
        """Overwrites the internal improper list with the improper
        defined in the supplied improper edge_index

        Parameters
        ----------
        edge_index:
            Edge index tensor of shape (4, n_impropers)
        """
        if edge_index.shape[0] != 4:
            raise ValueError(
                "improper edge index must have shape (4, n_impropers)"
            )

        self.impropers = tuple(edge_index.numpy().tolist())

    def to_mdtraj(self) -> mdtraj.Topology:
        """Convert to mdtraj format

        Returns
        -------
        topo:
            MDTraj topology instance from Topology
        """

        topo = mdtraj.Topology()
        chain = topo.add_chain()
        for i_at in range(self.n_atoms):
            if (
                self.names[i_at].strip().upper()
                not in Element._elements_by_symbol
            ):
                # TODO:change the default mass and radius to something more meaningful
                element = Element(
                    self.types[i_at], self.names[i_at], self.names[i_at], 10, 2
                )
            else:
                element = Element.getBySymbol(self.names[i_at])

            residue = topo.add_residue(self.resnames[i_at], chain)
            topo.add_atom(self.names[i_at], element, residue)
        for idx in range(len(self.bonds[0])):
            idx1, idx2 = self.bonds[0][idx], self.bonds[1][idx]
            a1, a2 = topo.atom(idx1), topo.atom(idx2)
            topo.add_bond(a1, a2)
        return topo

    @staticmethod
    def from_mdtraj(topology):
        """Build topology from an existing mdtraj topology.

        Parameters
        ----------
        topology:
            Input MDTraj topology

        Returns
        -------
        topo:
            Topology instance created from the input MDTraj topology
        """

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
    def from_ase(mol: Atoms, unique=True):
        """Build topology from an ASE Atoms instance

        Parameters
        ----------
        mol:
            ASE atoms instance
        unique:
            If True, only the unique bonds and angles will be added to the
            resulting Topology object. If False, all redundant (backwards)
            bonds and angles will be added as well.

        .. warning::
            The minimum image convention is applied to build the topology.

        Returns
        -------
        topo:
            Topology instance based on the ASE input
        """

        analysis = Analysis(mol)
        topo = Topology()
        types = mol.get_atomic_numbers()
        names = [ase_z2name[anum] for anum in types]
        for name, atom_type in zip(names, types):
            topo.add_atom(atom_type, name)

        if unique:
            bond_list = analysis.unique_bonds
        else:
            bond_list = analysis.all_bonds

        for atom, neighbors in enumerate(bond_list[0]):
            for bonded_neighbor in neighbors:
                topo.bonds[0].append(atom)
                topo.bonds[1].append(bonded_neighbor)
        if unique:
            angle_list = analysis.unique_angles
        else:
            angle_list = analysis.all_angles

        for atom, end_point_list in enumerate(angle_list[0]):
            for end_points in end_point_list:
                topo.angles[0].append(end_points[0])
                topo.angles[1].append(atom)
                topo.angles[2].append(end_points[1])
        return topo

    @staticmethod
    def from_file(filename: str):
        """Uses mdtraj reader to read the input topology."""
        topo = mdtraj.load(filename).topology
        return Topology.from_mdtraj(topo)

    def draw(
        self,
        layout: Callable = nx.drawing.layout.spring_layout,
        layout_kwargs: Dict = None,
        drawing_kwargs: Dict = None,
    ) -> None:
        """Use NetworkX to draw the current molecular topology.
        by default, node labels correspond to atom types.

        Parameters
        ----------
        layout:
            NetworkX layout drawing function (from networkx.drawing.layout) that
            determines the positions of the nodes
        layout_kwargs:
            keyword arguments for the node layout drawing function
        drawing_kwargs:
            keyword arguments for nx.draw
        """

        from matplotlib.pyplot import get_cmap

        if layout_kwargs == None:
            layout_kwargs = {}
        if drawing_kwargs == None:
            drawing_kwargs = {}
        connectivity = get_connectivity_matrix(self)
        graph = nx.Graph(connectivity.numpy())
        node_pos = layout(graph, **layout_kwargs)
        drawing_kwargs["pos"] = node_pos

        if "labels" not in list(drawing_kwargs.keys()):
            drawing_kwargs["labels"] = {
                node: str(self.types[node]) for node in graph.nodes
            }
        if "node_color" not in list(drawing_kwargs.keys()):
            num_colors = len(np.arange(1, max(self.types) + 2))
            cmap = get_cmap("viridis", num_colors)
            drawing_kwargs["node_color"] = [
                cmap.colors[node_type, :3] for node_type in self.types
            ]

        nx.draw(graph, **drawing_kwargs)


def get_connectivity_matrix(
    topology: Topology, directed: bool = False
) -> torch.tensor:
    """Produces a full connectivity matrix from the graph structure
    implied by Topology.bonds

    Parameters
    ----------
    topology:
        Topology for which a connectivity matrix will be constructed

    Returns
    -------
    connectivity_matrix:
        Torch tensor of shape (n_atoms, n_atoms) representing the
        connectivity/adjacency matrix from the bonded graph.
    directed:
        If True, an asymmetric connectivity matrix will be returned
        correspending to a directed graph. If false, the connectivity
        matrix will be symmetric and the corresponding graph will be
        undirected.
    """

    if len(topology.bonds[0]) == 0 and len(topology.bonds[1]) == 0:
        raise ValueError("No bonds in the topology.")
    if topology.n_atoms == 0:
        raise ValueError("n_atoms is not specified in the topology")

    connectivity_matrix = torch.zeros(topology.n_atoms, topology.n_atoms)
    bonds = topology.bonds2torch()
    connectivity_matrix[bonds[0, :], bonds[1, :]] = 1
    if directed == False:
        connectivity_matrix[bonds[1, :], bonds[0, :]] = 1

    return connectivity_matrix


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
        topology.add_dihedral(i, i + 1, i + 2, i + 3)


def get_n_pairs(
    connectivity_matrix: torch.Tensor, n: int = 3, unique: bool = True
) -> torch.tensor:
    """This function uses networkx to identify those pairs
    that are exactly n atoms away. Paths are found using Dijkstra's algorithm.

    Parameters
    ----------
    connectivity_matrix:
        Connectivity/adjacency matrix of the molecular graph of shape
        (n_atoms, n_atoms)
    n:
        Number of atoms to count away from the starting atom, with the starting
        atom counting as n=1
    unique:
        If True, the returned pairs will be unique and symmetrised.

    Returns
    -------
    pairs:
        Edge index tensor of shape (2, n_pairs)
    """
    graph = nx.Graph(connectivity_matrix.numpy())
    pairs = ([], [])
    for atom in graph.nodes:
        n_hop_paths = nx.single_source_dijkstra_path(graph, atom, cutoff=n)
        termini = [
            path[-1] for sub_atom, path in n_hop_paths.items() if len(path) == n
        ]
        for child_atom in termini:
            pairs[0].append(atom)
            pairs[1].append(child_atom)

    pairs = torch.tensor(pairs)
    if unique:
        pairs = _symmetrise_distance_interaction(pairs)
        pairs = torch.unique(pairs, dim=1)
    return pairs


def get_n_paths(connectivity_matrix, n=3, unique=True) -> torch.tensor:
    """This function use networkx to grab all connected paths defined
    by n connecting edges. Paths are found using Dijkstra's algorithm.

    Parameters
    ----------
    connectivity_matrix:
        Connectivity/adjacency matrix of the molecular graph of shape (n_atoms, n_atoms)
    n:
        Number of atoms to count away from the starting atom, with the starting atom counting as n=1
    unique:
        If True, the returned pairs will be unique and symmetrised such that the lower bead index precedes
        the higher bead index in each pair.

    Returns
    -------
    final_paths:
        Path index tensor of shape (n, n_pairs)
    """

    if n not in [2, 3, 4] and unique == True:
        raise NotImplementedError("Unique currently only works for n=2,3")

    graph = nx.Graph(connectivity_matrix.numpy())
    final_paths = [[] for i in range(n)]
    for atom in graph.nodes:
        n_hop_paths = nx.single_source_dijkstra_path(graph, atom, cutoff=n)
        paths = [path for _, path in n_hop_paths.items() if len(path) == n]
        # print(paths)
        for path in paths:
            # print(path)
            for k, sub_atom in enumerate(path):
                # print(sub_atom)
                final_paths[k].append(sub_atom)
    final_paths = torch.tensor(final_paths)
    if unique and n in [2, 3, 4]:
        final_paths = _symmetrise_map[n](final_paths)
        final_paths = torch.unique(final_paths, dim=1)

    return final_paths


def get_improper_paths(
    connectivity_matrix: torch.Tensor, unique: bool = True
) -> torch.tensor:
    """This function returns all paths defining an improper dihedral

            k
            |
        i - j - l

    where the order of connected nodes is given as [i,k,l,j] - i.e., the
    central node is reported last.

    Parameters
    ----------
    connectivity_matrix:
        Connectivity/adjacency matrix of the molecular graph of shape (n_atoms, n_atoms)
    unique:
        If True, the returned paths will be unique

    Returns
    -------
    final_paths:
        Path index tensor of shape (4, n_impropers)
    """

    n = 4
    neigh_counts = np.sum(connectivity_matrix.numpy(), axis=0)
    final_paths = [[] for i in range(n)]
    for i_nc, neigh_count in enumerate(neigh_counts):
        if neigh_count >= 3:
            neigh_list = np.where(connectivity_matrix.numpy()[i_nc] == 1)[0]
            for combo in combinations(neigh_list, 3):
                final_paths[-1].append(i_nc)
                for ic, ind in enumerate(combo):
                    final_paths[ic].append(ind)

    final_paths = torch.tensor(final_paths)
    if unique:
        final_paths = torch.unique(final_paths, dim=1)

    return final_paths

def _grab_improper_atom_groups(self, top: mdtraj.Topology, group=['CA','O','N','C'], 
    normal_idx=[0,3,4,2], gly_local_idx=[0,2,3,1]) -> List[Tuple]:
    '''
        Helper method to create neighborlist of atoms involved in gamma 2
            (C_{alpha}, O, N, C). Central atom comes last based on convention in internal_coordaintes
    '''
    group_list = []                                                               

    atom_list = [atom.name for atom in top.atoms]
    residues_list = [atom.residue.name for atom in top.atoms]
    for idx in range(len(atom_list) - 4):
        starting_res = residues_list[normal_idx[0] + idx]
        # Glycine handled separately since no CB
        if starting_res != "GLY":
            local_idx = normal_idx 
        else:
            local_idx = gly_local_idx 
        at1 = atom_list[idx + local_idx[0]]
        at2 = atom_list[idx + local_idx[1]]
        at3 = atom_list[idx + local_idx[2]]
        at4 = atom_list[idx + local_idx[3]]
        query_names = [at1, at2, at3, at4]
        if np.array_equal(query_names, group):
            group_list.append((idx + local_idx[0], idx+local_idx[1], idx+local_idx[2], idx+local_idx[3]))
    return group_list

def _grab_dihedral_atom_groups(self, top: mdtraj.Topology, phi=True, psi=True, omega=True):
    '''
        Helper function to select atoms involved in protein dihehdral backbone
        phi = [(C-1),N,CA,C]
        psi = [N,CA,C,(N+1)]
        omega = [(CA-1),(C-1),N,CA]
    '''
    dihedrals = []
    PHI_ATOMS = ["-C", "N", "CA", "C"]
    PSI_ATOMS = ["N", "CA", "C", "+N"]
    OMEGA_ATOMS = ["CA", "C", "+N", "+CA"]
    if phi: 
        phi_atoms = mdtraj.geometry.dihedral._atom_sequence(top,PHI_ATOMS)[1]        
    else: phi_atoms = None
    dihedrals.append(phi_atoms)

    if psi:
        psi_atoms = mdtraj.geometry.dihedral._atom_sequence(top,PSI_ATOMS)[1]        
    else: psi_atoms = None
    dihedrals.append(psi_atoms)

    if omega: 
        omega_atoms = mdtraj.geometry.dihedral._atom_sequence(top,OMEGA_ATOMS)[1]        
    else: omega_atoms = None
    dihedrals.append(omega_atoms)

    return dihedrals


def _dihedral_mapping_dictionary(self,dihedral_atoms, top) -> Dict:
    '''
        Helper function to assign each dihedral to a Amino Acid type
    '''
    from ..cg import CA_MAP
    dihedral_dictionary = {key[0]: [] for key in CA_MAP.keys()}
    resids = np.array([atom.residue.name for atom in top.atoms])
    for i in range(len(dihedral_atoms)):
        group = dihedral_atoms[i]
        res_group = resids[group]
        unique_res, counts = np.unique(res_group, return_counts=True)
        current_res = unique_res[np.argmax(counts)]
        dihedral_dictionary[current_res].append(group)
    for k,v in dihedral_dictionary.items():
        dihedral_dictionary[k] = np.array(v)
    return dihedral_dictionary