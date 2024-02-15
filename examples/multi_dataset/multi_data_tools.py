from typing import List, Tuple, Dict, Callable, Optional
import pickle
from tqdm import tqdm
import re
from mdtraj.core.topology import Atom
import numpy as np
import mdtraj as md
import os
from glob import glob
from cg_molecule import CGMolecule
from mlcg.geometry.topology import (
    Topology,
    get_connectivity_matrix,
    get_n_paths,
    get_n_pairs,
    get_improper_paths,
)
from mlcg.data.atomic_data import AtomicData
from mlcg.neighbor_list.neighbor_list import make_neighbor_list
import torch
from torch_geometric.loader import DataLoader
import networkx as nx
from mlcg.geometry._symmetrize import _symmetrise_distance_interaction
from networkx.algorithms.shortest_paths.unweighted import (
    bidirectional_shortest_path,
)
from mdtraj.core.topology import Atom, Residue
from mdtraj import Trajectory

from _embeddings import (
    embed2res,
    all_res,
    cb_types,
    opep_embedding_map,
    opep_termini_embedding_map,
    generate_embeddings,
)
import networkx as nx
from aggforce import linearmap as lm
from aggforce import agg as ag
from aggforce import constfinder as cf

from dataset_tools import (
    get_slice_cg_mapping,
)
from time import sleep


def natural_sort(name: str) -> str:
    """Performs natural sort"""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(name, key=alphanum_key)


def swap_pro_ca_cb(
    cg_atoms: List[Atom], cg_indices: List[int]
) -> Tuple[List[Atom], List[int]]:
    """Reverses PRO CA and CB positions in order to maintain sanity
    when dealing with PDBs (eg, opep pdbs) that for some reason see fit
    to reverse this order for PRO only.

    Parameters
    ----------
    cg_atoms:
        list of MDTraj Atom indices
    cg_indices:
        list of cg atom indices

    Returns
    -------
    swapped_atoms:
        List of MDTraj atom instances with the correct PRO CA/CB order
    swapped_cg_indices:
        List of cg indices with the correct PRO CA/CB order
    """
    assert len(cg_atoms) == len(cg_indices)
    swapped_atoms = []
    swapped_indices = []
    for n, (atom, idx) in enumerate(zip(cg_atoms, cg_indices)):
        if atom.residue.name == "PRO":
            if atom.name == "CB":
                swapped_atoms.append(
                    Atom(
                        "CA",
                        md.element.carbon,
                        cg_indices[n + 1],
                        atom.residue,
                    )
                )
                swapped_indices.append(cg_indices[n + 1])
            elif atom.name == "CA":
                swapped_atoms.append(
                    Atom(
                        "CB",
                        md.element.carbon,
                        cg_indices[n - 1],
                        atom.residue,
                    )
                )
                swapped_indices.append(cg_indices[n - 1])
            else:
                swapped_atoms.append(atom)
                swapped_indices.append(cg_indices[n])
        else:
            swapped_atoms.append(atom)
            swapped_indices.append(cg_indices[n])

    return swapped_atoms, swapped_indices


def get_mol_dictionary(
    pdb: Trajectory,
    pro_swap: bool = False,
    tag: str = "molecule",
    embedding_strategy: str = "opep",
) -> Dict:
    """Takes an all-atom pdb and maps it to the OPEPs CG mapping

    NOTE: ACE and NME/NHE caps are filtered automatically

    Parameters
    ----------
    pdb:
        MDTraj trajectory object
    pro_swap:
        If True, the positions of all PRO CAs and PRO CBs will
        be swapped.
    tag:
        String that identifies the protein
    embedding_strategy:
        String that determines whicch embedding strategy to use

    Returns
    -------
    mol_dictionary:
        dictionary of useful cg information
    """
    pdb = pdb.remove_solvent()
    if len([res for res in pdb.topology.residues if res.name == "NHE"]) == 0:
        # topology.select('protein') preserves ACE and NME but not NHE
        prot_idx = pdb.topology.select("protein")
        pdb = pdb.atom_slice(prot_idx)
    mol_dictionary = {}
    mol_dictionary["aa_traj"] = pdb

    # skip caps

    ACE_atoms = [
        atom for atom in pdb.topology.atoms if atom.residue.name == "ACE"
    ]
    NME_atoms = [
        atom for atom in pdb.topology.atoms if atom.residue.name == "NME"
    ]
    NHE_atoms = [
        atom for atom in pdb.topology.atoms if atom.residue.name == "NHE"
    ]

    base_residues = [
        res
        for res in pdb.topology.residues
        if res.name not in ["ACE", "NME", "NHE"]
    ]

    # Change NLE to LEU

    residues = []

    for residue in base_residues:
        if residue.name == "NLE":
            new_res = md.core.topology.Residue(
                "LEU",
                residue.index,
                residue.chain,
                residue.resSeq,
                residue.segment_id,
            )
            residues.append(new_res)
        else:
            residues.append(residue)

    glys = [res for res in residues if res.name == "GLY"]
    gly_idx = [i for i, res in enumerate(residues) if res.name == "GLY"]

    # get all the atoms within the residues
    num_all_atoms = 0
    for res in residues:
        num_all_atoms += len(list(res.atoms))

    # add in the ACE and NME/NHE atoms for all-atom checks in save_cg_coordforce
    num_all_atoms = (
        num_all_atoms + len(ACE_atoms) + len(NME_atoms) + len(NHE_atoms)
    )

    residue_names = [res.name for res in residues]
    # get cg atom indices
    cg_atoms_needed = ["N", "CA", "CB", "C", "O"]
    cg_atom_indices = []
    cg_md_atoms = []
    for idx, atom in enumerate(pdb.topology.atoms):
        # skip caps
        if atom.residue.name in ["ACE", "NME", "NHE"]:
            continue
        residue = atom.residue
        if residue.name == "NLE":  # NLE to LEU
            residue = md.core.topology.Residue(
                "LEU",
                residue.index,
                residue.chain,
                residue.resSeq,
                residue.segment_id,
            )
            atom.residue = residue
        atom_name = atom.name
        if residue.name in residue_names:
            if atom_name in cg_atoms_needed:
                cg_atom_indices.append(idx)
                cg_md_atoms.append(atom)
    if pro_swap == True:
        cg_md_atoms, cg_atom_indices = swap_pro_ca_cb(
            cg_md_atoms, cg_atom_indices
        )
    chainseq = [a.residue.chain.index for a in cg_md_atoms]
    # assemble molecular dictionary
    mol_dictionary["residues"] = residues
    mol_dictionary["chainseq"] = chainseq
    mol_dictionary["gly_idx"] = gly_idx
    mol_dictionary["cg_atoms"] = cg_md_atoms
    mol_dictionary["cg_indices"] = cg_atom_indices
    mol_dictionary["num_all_atoms"] = num_all_atoms

    # build CG types
    resseq = []
    res_idx = []
    for j, residue in enumerate(residues):
        if residue.name == "GLY":
            for _ in range(4):
                resseq.append(j + 1)
                res_idx.append(j)
        else:
            for _ in range(5):
                resseq.append(j + 1)
                res_idx.append(j)

    mol_dictionary["resmap"] = {
        k: v.name for k, v in zip(np.arange(1, len(residues) + 1), residues)
    }
    if len(list(pdb.topology.chains)) > 1:
        embedding_chain_kwarg = list(pdb.topology.chains)
    else:
        embedding_chain_kwarg = None
    types = generate_embeddings(
        embedding_strategy, residues, chains=embedding_chain_kwarg
    )
    mol_dictionary["resseq"] = resseq
    mol_dictionary["res_idx"] = res_idx
    assert len(types) == len(cg_atom_indices)
    assert len(resseq) == len(res_idx)
    mol_dictionary["types"] = types
    mol_dictionary["tag"] = tag

    return mol_dictionary


def save_cg_coordforce(
    loader: Callable,
    mol_dictionary: Dict,
    filenames: List[str],
    save_dir: str,
    only_coords: bool = False,
    mapping: str = "slice_aggregate",
) -> None:
    """Function for saving CG coordinates and forces

    Parameters
    ----------
    loader:
        python function that returns all-atom coordinates and forces given
        a file. Used for generalization between different all-atom data storage
        options.
    mol_dictionary:
        mol_dictionary object
    filenames:
        list of filenames or tuples of filenames for the all-atom data of
        the molecule specified by mol_dictionary
    save_dir:
        Directory in which CG coordinates and forces will be saved
    only_coords:
        If True, only coordinates will be loaded and saved. Useful for
        datasets where there is no force information (e.g., DESRES)
    mapping:
        Either "slice_aggregate" or "slice_optimize"
    """

    if mapping not in ["slice_aggregate", "slice_optimize"]:
        raise ValueError(
            "mapping {} is not 'slice_aggregate' nor 'slice_optimize'".format(
                mapping
            )
        )
    else:
        cg_map, _ = get_slice_cg_mapping(mol_dictionary["aa_traj"])

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if only_coords == True:
        aa_coord_list = []
        if len(filenames) == 1 and any([f.endswith(".h5") for f in filenames]):
            assert len(filenames) == 1
            coord, _ = loader(filenames[0], mol_dictionary["tag"])
            aa_coord_list.append(coord)
            del coord
            sleep(0.5)
        else:
            for fn in filenames:
                coord, _ = loader(fn, mol_dictionary["tag"])
                aa_coord_list.append(coord)
        aa_coords = np.concatenate(aa_coord_list)

        # Remove trouble frame
        if mol_dictionary["tag"] == "opep_1008":
            print(
                "removing trouble frame for {}...".format(mol_dictionary["tag"])
            )
            pre_len = len(aa_coords)
            trouble_frame = 17394
            aa_coords = np.delete(aa_coords, trouble_frame, axis=0)
            assert len(aa_coords) == pre_len - 1

        constraints = cf.guess_pairwise_constraints(
            aa_coords[::100], threshold=5e-3
        )
        n_fg_sites = aa_coords.shape[1]
        config_map = lm.LinearMap(cg_map)
        config_map_matrix = config_map.standard_matrix

        cg_coords = config_map_matrix @ aa_coords
        cg_topo = CGMolecule(
            names=[atom.name for atom in mol_dictionary["cg_atoms"]],
            resseq=mol_dictionary["resseq"],
            chainseq=mol_dictionary["chainseq"],
            resmap=mol_dictionary["resmap"],
            bonds="standard",
        )
        num_atoms = cg_coords.shape[1]

        # PDB is saved in NANOMETERS (for molecular viewer conventions)
        # coordinates are saved in ANGSTROMS
        # forces are saved in KCAL/MOL/ANGSTROM
        cg_traj = cg_topo.make_trajectory(
            cg_coord[0].reshape(1, num_atoms, 3) / 10.0
        )
        cg_traj.save_pdb(
            save_dir + "/{}_cg_structure.pdb".format(mol_dictionary["tag"])
        )
        np.save(
            save_dir + "{}_cg_coords.npy".format(mol_dictionary["tag"]),
            cg_coords,
        )
        np.save(
            save_dir + "{}_cg_embeds.npy".format(mol_dictionary["tag"]),
            mol_dictionary["types"],
        )
        np.save(
            save_dir + "{}_cg_map.npy".format(mol_dictionary["tag"]),
            config_map_matrix,
        )

    if only_coords == False:
        aa_coord_list = []
        aa_force_list = []
        # load the files, checking against the mol dictionary
        if len(filenames) == 1 and any([f.endswith(".h5") for f in filenames]):
            assert len(filenames) == 1
            coord, force = loader(filenames[0], mol_dictionary["tag"])
            assert coord.shape == force.shape
            aa_coord_list.append(coord)
            aa_force_list.append(force)
            del coord, force
        else:
            for fn in filenames:
                coord, force = loader(fn, mol_dictionary["tag"])
                assert coord.shape == force.shape
                aa_coord_list.append(coord)
                aa_force_list.append(force)
        aa_coords = np.concatenate(aa_coord_list)
        aa_forces = np.concatenate(aa_force_list)
        with open(
            save_dir + "/{}_mol_dict.pkl".format(mol_dictionary["tag"]), "wb"
        ) as pfile:
            pickle.dump(mol_dictionary, pfile)

        # Remove trouble frame
        if mol_dictionary["tag"] == "opep_1008":
            print(
                "removing trouble frame for {}...".format(mol_dictionary["tag"])
            )
            pre_len = len(aa_coords)
            trouble_frame = 17394
            aa_coords = np.delete(aa_coords, trouble_frame, axis=0)
            aa_forces = np.delete(aa_forces, trouble_frame, axis=0)
            assert len(aa_coords) == len(aa_forces) == pre_len - 1

        constraints = cf.guess_pairwise_constraints(
            aa_coords[::100], threshold=5e-3
        )

        n_fg_sites = aa_coords.shape[1]
        config_map = lm.LinearMap(cg_map)
        config_map_matrix = config_map.standard_matrix
        if mapping == "slice_aggregate":
            method = lm.constraint_aware_uni_map
            force_stride = 1
            force_agg_results = ag.project_forces(
                xyz=None,
                forces=aa_forces[::force_stride],
                config_mapping=config_map,
                constrained_inds=constraints,
                method=method,
            )

        elif mapping == "slice_optimize":
            method = lm.qp_linear_map
            if mol_dictionary["tag"].startswith("opep"):
                force_stride = 1
            if mol_dictionary["tag"].startswith(""):
                force_stride = 1
            if mol_dictionary["tag"].startswith("cath"):
                force_stride = 1
            l2 = 1e3
            force_agg_results = ag.project_forces(
                xyz=None,
                forces=aa_forces[::force_stride],
                config_mapping=config_map,
                constrained_inds=constraints,
                method=method,
                l2_regularization=l2,
            )

        else:
            raise RuntimeError(
                "Uh-oh: mapping {} is neither 'slice_aggregate' nor 'slice_optimize'.".format(
                    mapping
                )
            )

        force_map_matrix = force_agg_results["map"].standard_matrix

        cg_coords = config_map_matrix @ aa_coords
        cg_forces = force_map_matrix @ aa_forces

        cg_topo = CGMolecule(
            names=[atom.name for atom in mol_dictionary["cg_atoms"]],
            resseq=mol_dictionary["resseq"],
            chainseq=mol_dictionary["chainseq"],
            resmap=mol_dictionary["resmap"],
            bonds="standard",
        )
        num_atoms = cg_coords.shape[1]

        # PDB is saved in NANOMETERS (for molecular viewer conventions)
        # coordinates are saved in ANGSTROMS
        # forces are saved in KCAL/MOL/ANGSTROM
        cg_traj = cg_topo.make_trajectory(
            cg_coords[0].reshape(1, num_atoms, 3) / 10.0
        )
        cg_traj.save_pdb(
            save_dir + "/{}_cg_structure.pdb".format(mol_dictionary["tag"])
        )
        np.save(
            save_dir + "{}_cg_coords.npy".format(mol_dictionary["tag"]),
            cg_coords,
        )
        np.save(
            save_dir + "{}_cg_forces.npy".format(mol_dictionary["tag"]),
            cg_forces,
        )
        np.save(
            save_dir + "{}_cg_embeds.npy".format(mol_dictionary["tag"]),
            mol_dictionary["types"],
        )
        np.save(
            save_dir + "{}_cg_map.npy".format(mol_dictionary["tag"]),
            config_map_matrix,
        )
        np.save(
            save_dir + "{}_cg_force_map.npy".format(mol_dictionary["tag"]),
            force_map_matrix,
        )


def get_termini_atoms(md_topo: md.Topology) -> Tuple[List[int], List[int]]:
    """Returns termini atoms, assuming an OPEP CG mapping. If there are
    multiple molecules, then the termini atoms are found and reported for
    each.

    Parameters
    ----------
    md_topo:
        CG MDtraj topology

    Returns
    -------
    n_term_atoms:
        List of N terminus CG indices
    c_term_atoms:
        List of C Terminus CG indices
    """
    chains = list(md_topo.chains)
    num_gly_n_term = 0  # for asserts
    num_gly_c_term = 0  # for asserts
    n_term_atoms = []
    chain_accum = (
        0  # measures how many atoms we have passed after each successive chain
    )
    for chain in chains:
        # If the molecule is a monopeptide, it will just have "bulk" atoms
        if len(list(chain.residues)) == 1:
            continue
        if list(chain.residues)[0].name == "GLY":
            num_gly_n_term += 1
        for i, atom in enumerate(chain.atoms):
            if atom.name == "O":
                n_term_atoms.append(i + chain_accum)
                break
            else:
                n_term_atoms.append(i + chain_accum)
        chain_accum += len(list(chain.atoms))

    # c_term_atoms (N, CA, CB, C, O)
    chain_accum = (
        0  # measures how many atoms we have passed after each successive chain
    )
    c_term_atoms = []
    for chain in chains:
        # If the molecule is a monopeptide, it will just have "bulk" atoms
        if len(list(chain.residues)) == 1:
            continue
        if list(chain.residues)[-1].name == "GLY":
            num_gly_c_term += 1
        # generator objects are not reversible :P
        for i, atom in enumerate(reversed(list(chain.atoms))):
            if atom.name == "N":
                c_term_atoms.append(
                    len(list(chain.atoms)) - i - 1 + chain_accum
                )
                break
            else:
                c_term_atoms.append(
                    len(list(chain.atoms)) - i - 1 + chain_accum
                )
        chain_accum += len(list(chain.atoms))
    c_term_atoms = sorted(c_term_atoms)
    if len(n_term_atoms) != 0:
        assert len(n_term_atoms) == 5 * len(chains) - num_gly_n_term
    if len(c_term_atoms) != 0:
        assert len(c_term_atoms) == 5 * len(chains) - num_gly_c_term
    return n_term_atoms, c_term_atoms


def make_mlcg_topo(
    md_topo: md.Topology, types: List[int]
) -> Tuple[Topology, torch.Tensor, torch.Tensor]:
    """Makes MLCG topology with bond and angle edges

    Parameters
    ----------
    md_topo:
        CG MDtraj topology
    types:
        list of CG types

    Returns
    -------
    mlcg_topo:
        MLCG CG topology
    bond_edges:
        edge tensor of bonds
    angle_edges:
        edge tensor of angles
    """
    mlcg_topo = Topology.from_mdtraj(md_topo)
    mlcg_topo.types = types
    conn_mat = get_connectivity_matrix(mlcg_topo)

    # Get full bond/angle sets
    bond_edges = get_n_paths(conn_mat, n=2).numpy()
    angle_edges = get_n_paths(conn_mat, n=3).numpy()
    # Add bonds/angles to topology
    mlcg_topo.bonds_from_edge_index(torch.tensor(bond_edges))
    mlcg_topo.angles_from_edge_index(torch.tensor(angle_edges))
    return mlcg_topo, bond_edges, angle_edges


def isolate_bonds(
    target_atoms: List[List[int]], full_bond_set: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Creates bond sublists from a larger bond list

    Parameters
    ----------
    target_atoms:
        if a bond atom from full_bond_set is in this list, that corresponding bond
        is moved to the isolated bond list. If neither bond atom is in this list,
        then the corresponding bond remains in the bulk bond list
    full_bond_set:
        The set of full bonds that is checked against the target_atoms

    Returns
    -------
    isolated_bonds:
        the bonds that have been isolated, where each islated bond contains at least
        one atom in the target_atoms list
    bulk_bonds:
        The bonds that have not be islolated, because each bond in this set does not
        contain any atoms in the target_atoms list
    """

    isolated_bonds = [([], []) for _ in range(len(target_atoms))]
    bulk_bonds = ([], [])
    for i in range(full_bond_set.shape[1]):
        bulk_bond = True
        for j, target_list in enumerate(target_atoms):
            b1, b2 = full_bond_set[0, i], full_bond_set[1, i]
            if b1 in target_list or b2 in target_list:
                isolated_bonds[j][0].append(b1)
                isolated_bonds[j][1].append(b2)
                bulk_bond = False
        if bulk_bond == True:
            bulk_bonds[0].append(b1)
            bulk_bonds[1].append(b2)
    bulk_bonds = np.array(bulk_bonds)
    return isolated_bonds, bulk_bonds


def isolate_angles(
    target_atoms: List[List[int]], full_angle_set: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Creates angle sublists from a larger angle list

    Parameters
    ----------
    target_atoms:
        if a angle atom from full_angle_set is in this list, that corresponding angle
        is moved to the isolated angle list. If neither angle atom is in this list,
        then the corresponding angle remains in the bulk angle list
    full_angle_set:
        The set of full angles that is checked against the target_atoms

    Returns
    -------
    isolated_angles:
        the angles that have been isolated, where each isolated angle contains at least
        one atom in the target_atoms list
    bulk_angles:
        The angles that have not be islolated, because each angle in this set does not
        contain any atoms in the target_atoms list
    """
    isolated_angles = [([], [], []) for _ in range(len(target_atoms))]
    bulk_angles = ([], [], [])
    for i in range(full_angle_set.shape[1]):
        bulk_angle = True
        for j, target_list in enumerate(target_atoms):
            b1, b2, b3 = (
                full_angle_set[0, i],
                full_angle_set[1, i],
                full_angle_set[2, i],
            )
            if b1 in target_list or b2 in target_list or b3 in target_list:
                isolated_angles[j][0].append(b1)
                isolated_angles[j][1].append(b2)
                isolated_angles[j][2].append(b3)
                bulk_angle = False
        if bulk_angle == True:
            bulk_angles[0].append(b1)
            bulk_angles[1].append(b2)
            bulk_angles[2].append(b3)
    bulk_angles = np.array(bulk_angles)
    return isolated_angles, bulk_angles


def get_pseudobond_edges(
    bond_types: List[List[int]],
    atom_types: List[int],
    connectivity_matrix: np.array,
    separation: int,
) -> np.ndarray:
    pairs = get_n_pairs(connectivity_matrix, n=separation)
    pbonds = ([], [])
    for j in range(len(bond_types)):
        t1, t2 = bond_types[j][0], bond_types[j][1]
        for i in range(pairs.shape[1]):
            a1, a2 = pairs[0, i], pairs[1, i]
            if atom_types[a1] == t1 and atom_types[a2] == t2:
                pbonds[0].append(a1)
                pbonds[1].append(a2)
            elif atom_types[a1] == t2 and atom_types[a2] == t1:
                pbonds[0].append(a1)
                pbonds[1].append(a2)
            else:
                continue
    # sort based on the first bond indices
    pbonds = np.array(pbonds).T
    pbonds = sorted(pbonds, key=lambda x: x[0])
    return np.array(pbonds).T


def get_pseudobonds(
    num_res: int, topo: Topology, strategy: str = "standard"
) -> List[Tuple]:
    """Gets pseudobonds for OPEP mapping

    Parameters
    ----------
    num_res:
        number of residues in the CG molecule
    topo:
        CG MLCG topology instance
    strategy:
        list of lists of strings of the form:

            ["CGATOM1_CGATOM2_separation"]

        where CGATOM1 is the first CG atom, CGATOM2 is
        the second CG atom, and separation is an integer
        equal to the number of edges +1 separating the two
        CG atoms in the bonded CF graph. Eg, to get all
        neighboring CA-CA pseudobonds, one would specify:

            ["CA_CA_4"]

    Returns
    -------
    pseudobonds:
        list of tuples of the form:

            ("tag", edges)
    """
    conn_mat = get_connectivity_matrix(topo)
    pseudobonds = []
    for spec in strategy:
        atom1, atom2, separation = spec.split("_")
        separation = int(separation)
        bond_types = [[opep_embedding_map[atom1], opep_embedding_map[atom2]]]
        # accounting for GLY CA embedding
        if atom1 == "CA" and atom2 != "CA":
            bond_types += [
                [opep_embedding_map["GLY"], opep_embedding_map[atom2]]
            ]
        if atom2 == "CA" and atom1 != "CA":
            bond_types += [
                [opep_embedding_map[atom1], opep_embedding_map["GLY"]]
            ]
        if atom1 == "CA" and atom2 == "CA":
            bond_types += [
                [opep_embedding_map["GLY"], opep_embedding_map["GLY"]]
            ]
            bond_types += [
                [opep_embedding_map["GLY"], opep_embedding_map["CA"]]
            ]
        sub_edges = get_pseudobond_edges(
            bond_types, topo.types, conn_mat, separation
        )
        pbond_tag = atom1 + "_" + atom2 + "_pbonds"
        pseudobonds.append((pbond_tag, sub_edges))

    return pseudobonds


def get_non_bonded_set(
    topo: Topology,
    bond_edges: torch.Tensor,
    angle_edges: torch.Tensor,
    pbond_edges: Optional[List[Tuple]] = None,
    res_idx: List[int] = None,
    min_pair: int = 4,
    res_exclusion: int = 1,
) -> np.array:
    """Returns the non-bonded set

    Parameters
    ----------
    topo:
        mlcg topology
    bond_edges:
        Edge index for CG bonds
    angle_edges:
        Edge index for CG angles
    pbond_edges:
        Edge index for CG pseudobonds that are removed from the non-bonded set
    res_idx:
        List of zero-based residue indices for each atom.
    min_pair:
        Minimum number of bond edges between two atoms in order to be considered
        a member of the non-bonded set
    res_exclusion:
        If supplied, pairs within res_exclusion residues of each other are removed
        from the non-bonded set

    Returns
    -------
    non_bonded_edges
    """
    fully_connected_edges = _symmetrise_distance_interaction(
        topo.fully_connected2torch()
    ).numpy()
    conn_mat = get_connectivity_matrix(topo).numpy()
    graph = nx.Graph(conn_mat)
    non_bonded_edges = ([], [])
    for i in range(fully_connected_edges.shape[1]):
        bonded = False
        edge = tuple(fully_connected_edges[:, i])

        for j in range(bond_edges.shape[1]):
            if edge == tuple(bond_edges[:, j]):
                bonded = True

        if not isinstance(pbond_edges, type(None)):
            for j in range(pbond_edges.shape[1]):
                if edge == tuple(pbond_edges[:, j]):
                    bonded = True

        for j in range(angle_edges.shape[1]):
            if edge == tuple(angle_edges[[0, 2], j]):
                bonded = True

        try:
            if (
                len(bidirectional_shortest_path(graph, edge[0], edge[1]))
                < min_pair
            ):
                bonded = True
        except nx.exception.NetworkXNoPath:
            bonded = False

        if np.abs(res_idx[edge[0]] - res_idx[edge[1]]) < res_exclusion:
            bonded = True

        if bonded == False:
            non_bonded_edges[0].append(edge[0])
            non_bonded_edges[1].append(edge[1])

    non_bonded_edges = torch.tensor(non_bonded_edges)
    non_bonded_edges = torch.unique(
        _symmetrise_distance_interaction(non_bonded_edges), dim=1
    ).numpy()
    return non_bonded_edges


def get_improper_atoms(
    connectivity_matrix: torch.Tensor,
    atom_types: np.ndarray,
    dihedral_type: str,
) -> np.ndarray:
    """Helper method to grab atoms involved in the
    planes defining molecular dihedrals

    Parameters
    ----------
    connectivity_matrix:
        MDTraj CG topology
    atom_types:
        Array of atom types for the molecule
    dihedral_types:
        String specifying if the dihedral is of type
        gamma_1 or gamma_2

    Returns
    -------
    improper_atoms:
        Numpy array of shape (num_dihedrals, 4),
        containing the improper dihedral atoms
    """

    if dihedral_type not in ["gamma_1", "gamma_2"]:
        raise ValueError(
            'For improper dihedrals, dihedral types must either be "gamma_1" or "gamma_2"; supplied type "{}" is invalid.'.format(
                dihedral_type
            )
        )
    all_impropers = get_improper_paths(connectivity_matrix).t()
    needed_impropers = []
    if dihedral_type == "gamma_1":
        for improper in all_impropers:
            improper_types = atom_types[improper.numpy()]
            if any(
                [
                    cb_type in improper_types
                    for cb_type in cb_types
                    if cb_type != opep_embedding_map["GLY"]
                ]
            ):
                needed_impropers.append(improper.numpy())

    if dihedral_type == "gamma_2":
        # enumerate oxygen types
        oxygen_types = [
            opep_embedding_map["O"],
            opep_termini_embedding_map["C_O"],
        ]
        for improper in all_impropers:
            improper_types = atom_types[improper.numpy()]
            if any([o_type in improper_types for o_type in oxygen_types]):
                needed_impropers.append(improper.numpy())
    if len(needed_impropers) > 0:
        needed_impropers = np.stack(needed_impropers)
    else:
        needed_impropers = np.zeros((0, 4))
    return needed_impropers


def isolate_pro_dihedral_sets(
    top: md.Topology, groups: np.ndarray, dihedral_type: str
) -> List[Tuple[str, torch.Tensor]]:
    """Helper function to separate PRO and non PRO dihedral angles. Meant
    to be used with omega and gamma1 angles, where PRO dependence is non-trivial.

    Parameters
    ----------
    topo:
        CG MDTraj topology
    atom_groups:
        List of 4 atom quads defining dihedrals
    dihedral_type:
        Tag denoting the type of dihedral

    Returns
    -------
    edges_with_tags:
    """

    dihedral_dictionary = {}
    dihedrals = []
    if dihedral_type not in ["omega"]:
        raise ValueError(
            "dihedral type {} is not 'omega'.".format(dihedral_type)
        )
    # First, grab the PRO residue indices:
    pro_indices = []
    non_pro_indices = []
    for ids in groups:
        if top.atom(ids[-1]).residue.code == "P":
            pro_indices.append(ids)
        else:
            non_pro_indices.append(ids)
    pro_indices = np.array(pro_indices)
    non_pro_indices = np.array(non_pro_indices)
    dihedrals.append(("pro_" + dihedral_type, torch.tensor(pro_indices.T)))
    dihedrals.append(
        ("non_pro_" + dihedral_type, torch.tensor(non_pro_indices.T))
    )

    return dihedrals


def legacy_isolate_pro_dihedral_sets(
    top: md.Topology, groups: np.ndarray, dihedral_type: str
) -> List[Tuple[str, torch.Tensor]]:
    """Helper function to separate PRO and non PRO dihedral angles. Meant
    to be used with omega and gamma1 angles, where PRO dependence is non-trivial.

    ..warn:

        THIS IS A LEGACY METHOD. PLEASE USE `isolate_pro_dihedral_sets` when
        building new models.

    Parameters
    ----------
    topo:
        CG MDTraj topology
    atom_groups:
        List of 4 atom quads defining dihedrals
    dihedral_type:
        Tag denoting the type of dihedral

    Returns
    -------
    edges_with_tags:
    """

    dihedral_dictionary = {}
    dihedrals = []
    if dihedral_type not in ["omega"]:
        raise ValueError(
            "dihedral type {} is not 'omega'.".format(dihedral_type)
        )
    # First, grab the PRO residue indices:
    pro_indices = []
    non_pro_indices = []
    for k, resi in enumerate(top.residues):
        if resi.code == "P":
            pro_indices.append(k)
        else:
            non_pro_indices.append(k)
    pro_omega_indices = (
        np.array(pro_indices) - 1
    )  # omega angles start in previous residue
    non_pro_omega_indices = (
        np.array(non_pro_indices) - 1
    )  # omega angles start in previous residue

    if len(pro_omega_indices) > 0:
        dihedral_dictionary["pro"] = groups[pro_omega_indices]
    else:
        dihedral_dictionary["pro"] = np.array([[]])
    dihedral_dictionary["non_pro"] = groups[non_pro_omega_indices]

    for name in dihedral_dictionary.keys():
        tag = name + "_" + dihedral_type
        dihedrals.append((tag, torch.tensor(dihedral_dictionary[name].T)))
    return dihedrals


def get_phi_psi_sets(
    top: md.Topology,
    atom_groups: np.ndarray,
    dihedral_type: str,
) -> List[Tuple]:
    """Helper method to retrieve dihedral sets based on
    amino acid identity. Meant to be used with Phi and Psi angles.

    Parameters
    ----------
    topo:
        CG MDTraj topology
    atom_groups:
        List of 4 atom quads defining dihedrals
    dihedral_type:
        Tag denoting the type of dihedral

    Returns
    -------
    edges_with_tags:
    """

    dihedrals = []
    dihedral_dictionary = {resname: [] for resname in all_res}
    resids = np.array([str(atom.residue.name) for atom in top.atoms])
    for i in range(len(atom_groups)):
        group = atom_groups[i]
        res_group = resids[group]
        unique_res, counts = np.unique(
            res_group, return_counts=True
        )  # magic to get majority residue id
        current_res = unique_res[np.argmax(counts)]
        dihedral_dictionary[current_res].append(group)

    for k, v in dihedral_dictionary.items():
        dihedral_dictionary[k] = np.array(v)

    for name in all_res:
        tag = name + "_" + dihedral_type
        dihedrals.append((tag, torch.tensor(dihedral_dictionary[name].T)))
    return dihedrals


def get_gamma_sets(
    atom_groups: np.ndarray,
    dihedral_type: str,
) -> List[Tuple]:
    """Helper method to retrieve dihedral sets based on
    amino acid identity. Meant to be used with gamma angles.

    Parameters
    ----------
    atom_groups:
        List of 4 atom quads defining dihedrals
    dihedral_type:
        Tag denoting the type of dihedral

    Returns
    -------
    edges_with_tags:
    """
    if dihedral_type not in ["gamma_1", "gamma_2"]:
        raise ValueError(
            "dihedral type {} is neither 'gamma_1' nor 'gamma_2.".format(
                dihedral_type
            )
        )

    dihedral_dictionary = {}
    dihedrals = []

    dihedral_dictionary["all_residues"] = (
        atom_groups  # imbue each residue with all atom groups
    )
    dihedrals.append((dihedral_type, torch.tensor(atom_groups.T)))

    return dihedrals


def get_edges_and_orders(
    md_topo: Topology,
    bond_edges: torch.Tensor,
    angle_edges: torch.Tensor,
    pseudobonds: Tuple[List] = None,
    dihedrals: Tuple[List] = None,
    non_bonded_edges: Optional[torch.Tensor] = None,
    termini_stats: bool = True,
) -> List[Tuple[str, int, torch.Tensor]]:
    """Generates tagged and ordered edges

    Parameters
    ----------
    md_topo:
        CG MDTraj instance
    bond_edges:
        bond edges
    angle_edges:
        angle_edges
    pseudobonds:
        pseudobond edges with tags
    dihedrals:
        dihedral edges with tags
    non_bonded_edges:
        non-bonded edges

    termini_stats:
        If True, all bond/angle/pseudobond edges will be further split into
        N terminus, C terminus, and bulk sets

    Returns
    -------
    edges_and_orders:
        list of tuples of the form:

            ("edge_tag", order, edges)
    """
    edges_and_orders = []
    if termini_stats == True:
        # Remove terminal features from full sets
        # isolate termini atoms
        # N terminus (N, CA, CB, C, O)
        n_term_atoms, c_term_atoms = get_termini_atoms(md_topo)
        term_bonds, bulk_bonds = isolate_bonds(
            [n_term_atoms, c_term_atoms], bond_edges
        )
        n_term_bonds = np.array(term_bonds[0])
        c_term_bonds = np.array(term_bonds[1])
        edges_and_orders.append(("n_term_bonds", 2, n_term_bonds))
        edges_and_orders.append(("bulk_bonds", 2, bulk_bonds))
        edges_and_orders.append(("c_term_bonds", 2, c_term_bonds))

        term_angles, bulk_angles = isolate_angles(
            [n_term_atoms, c_term_atoms], angle_edges
        )
        n_term_angles = np.array(term_angles[0])
        c_term_angles = np.array(term_angles[1])
        edges_and_orders.append(("n_term_angles", 3, n_term_angles))
        edges_and_orders.append(("bulk_angles", 3, bulk_angles))
        edges_and_orders.append(("c_term_angles", 3, c_term_angles))

        if pseudobonds != None:
            bulk_pbond_list = []
            n_term_pbond_list = []
            c_term_pbond_list = []
            for pbond in pseudobonds:
                pbond_tag, pbond_edges = pbond
                term_pbonds, bulk_pbonds = isolate_bonds(
                    [n_term_atoms, c_term_atoms], pbond_edges
                )
                n_term_pbonds = np.array(term_pbonds[0])
                c_term_pbonds = np.array(term_pbonds[1])
                n_term_pbond_list.append(
                    ("n_term_" + pbond_tag, 2, n_term_pbonds)
                )
                bulk_pbond_list.append(("bulk_" + pbond_tag, 2, bulk_pbonds))
                c_term_pbond_list.append(
                    ("c_term_" + pbond_tag, 2, c_term_pbonds)
                )
            edges_and_orders += n_term_pbond_list
            edges_and_orders += bulk_pbond_list
            edges_and_orders += c_term_pbond_list

        if dihedrals != None:
            for dihedral in dihedrals:
                dihedral_tag, dihedral_edges = dihedral
                if len(dihedral_edges) == 0:
                    dihedral_edges = torch.tensor([]).reshape(4, 0)
                edges_and_orders.append((dihedral_tag, 4, dihedral_edges))

        edges_and_orders.append(("non_bonded", 2, non_bonded_edges))
    else:
        edges_and_orders.append(("bonds", 2, bond_edges))
        edges_and_orders.append(("angles", 3, angle_edges))
        if pseudobonds != None:
            for pbond in pseudobonds:
                pbond_tag, pbond_edges = pbond
                edges_and_orders.append((pbond_tag, 2, pbond_edges))
        if dihedrals != None:
            for dihedral in dihedrals:
                dihedral_tag, dihedral_edges = dihedral
                if len(dihedral_edges) == 0:
                    dihedral_edges = torch.tensor([]).reshape(4, 0)
                edges_and_orders.append((dihedral_tag, 4, dihedral_edges))
        if len(non_bonded_edges) > 0:
            edges_and_orders.append(("non_bonded", 2, non_bonded_edges))
    return edges_and_orders


def get_atomic_data_list(
    coords: np.array,
    types: List[int],
    edges_and_orders: List[Tuple[str, int, torch.Tensor]],
) -> Tuple[List[AtomicData], Dict]:
    """makes a list of AtomicData instances

    Parameters
    ----------
    coords:
        Numpy array of CG coordinates of shape (n_frames, n_atoms, 3)
    types:
        Numpy array of CG types of shape (n_atoms)
    edges_and_orders:
        list of tuples of the form:

            ("tag", order, edges)

    Returns
    -------
    sub_data_list:
        List of AtomicData instances
    prior_nls:
        Dictionary of neighborlists for each prior
    """
    sub_data_list = []
    tags = [x[0] for x in edges_and_orders]
    orders = [x[1] for x in edges_and_orders]
    edges = [
        torch.tensor(x[2]).type(torch.LongTensor) for x in edges_and_orders
    ]
    assert len(tags) == len(orders) == len(edges)
    # load coordinates - stride by 10 or prepare for your RAM to suffer
    prior_nls = {
        tag: make_neighbor_list(tag, order, edge)
        for tag, order, edge in zip(tags, orders, edges)
    }
    for i in range(len(coords)):
        data = AtomicData.from_points(
            pos=torch.tensor(coords[i]),
            atom_types=torch.tensor(types),
            masses=None,
            neighborlist=prior_nls,
        )
        sub_data_list.append(data)
    return sub_data_list, prior_nls


def get_mlcg_topo(
    md_topo: md.Topology, types: List[int]
) -> Tuple[Topology, torch.Tensor, torch.Tensor]:
    """Makes MLCG topology with bond and angle edges

    Parameters
    ----------
    md_topo:
        CG MDtraj topology

    Returns
    -------
    mlcg_topo:
        MLCG CG topology
    bond_edges:
        edge tensor of bonds
    angle_edges:
        edge tensor of angles
    """
    mlcg_topo = Topology.from_mdtraj(md_topo)
    mlcg_topo.types = types
    conn_mat = get_connectivity_matrix(mlcg_topo)

    # Get full bond/angle sets
    bond_edges = get_n_paths(conn_mat, n=2).numpy()
    angle_edges = get_n_paths(conn_mat, n=3).numpy()
    # Add bonds/angles to topology
    mlcg_topo.bonds_from_edge_index(torch.tensor(bond_edges))
    mlcg_topo.angles_from_edge_index(torch.tensor(angle_edges))
    return mlcg_topo, bond_edges, angle_edges


def process_accumulation(
    mol_dictionary: Dict,
    data_generation_dict: Dict,
    stride=100,
) -> List[AtomicData]:
    """Processes data accumulation for CG data before collation.
    This is the main loop associated with aggregating molecular
    data just before fitting any priors.

    Parameters
    ----------
    mol_dictionary:
        CG molecule dictionary
    data_generation_dict:
        data generation dictionary
    stride:
        determines the stride to apply to stored coordinates

    Returns
    -------
    sub_data_list:
        List of AtomicData instances according to the specified
        prior neighborlists for a single molecule
    """
    types = mol_dictionary["types"]
    res_idx = mol_dictionary["res_idx"]

    traj = md.load(
        data_generation_dict["base_save_dir"]
        + mol_dictionary["tag"]
        + "_cg_structure.pdb"
    )
    assert all(
        [
            res1.name == res2.name
            for res1, res2 in zip(
                mol_dictionary["residues"], list(traj.topology.residues)
            )
        ]
    )
    # assert len(mol_dictionary["cg_atoms"]) == len(
    #    list(alt_cg_traj.topology.atoms)
    # )

    md_topo = traj.topology

    # Make MLCG topology
    residue_list = list([res.name for res in md_topo.residues])
    num_res = len(residue_list)
    num_gly = residue_list.count("GLY")
    mlcg_topo, bond_edges, angle_edges = get_mlcg_topo(md_topo, types)
    conn_mat = get_connectivity_matrix(mlcg_topo)
    num_molecules = nx.number_connected_components(nx.Graph(conn_mat.numpy()))

    if data_generation_dict["pseudobonds"] != None:
        pseudobonds = get_pseudobonds(
            num_res, mlcg_topo, strategy=data_generation_dict["pseudobonds"]
        )
        all_pbond_edges = np.concatenate(
            [edge_list for _, edge_list in pseudobonds], axis=1
        )
    else:
        pseudobonds = None
        all_pbond_edges = None

    # Construct the non-bonded set
    non_bonded_edges = get_non_bonded_set(
        mlcg_topo,
        bond_edges,
        angle_edges,
        all_pbond_edges,
        res_idx,
        min_pair=data_generation_dict["non_bonded_min"],
        res_exclusion=data_generation_dict["non_bonded_res_exclusion"],
    )

    if data_generation_dict["dihedrals"] != None:
        all_dihedrals = []
        for dihedral_type in data_generation_dict["dihedrals"]:
            if dihedral_type == "phi":
                atom_groups, _ = md.compute_phi(traj)
                dihedrals = get_phi_psi_sets(
                    md_topo,
                    atom_groups,
                    dihedral_type,
                )
            if dihedral_type == "psi":
                atom_groups, _ = md.compute_psi(traj)
                dihedrals = get_phi_psi_sets(
                    md_topo,
                    atom_groups,
                    dihedral_type,
                )
            if dihedral_type == "omega":
                atom_groups, _ = md.compute_omega(traj)
                dihedrals = isolate_pro_dihedral_sets(
                    md_topo, atom_groups, dihedral_type
                )
            if dihedral_type == "gamma_1":
                atom_groups = get_improper_atoms(
                    conn_mat, np.array(types), "gamma_1"
                )
                dihedrals = get_gamma_sets(atom_groups, dihedral_type)
                assert len(atom_groups) == num_res - num_gly
            if dihedral_type == "gamma_2":
                atom_groups = get_improper_atoms(
                    conn_mat, np.array(types), "gamma_2"
                )
                assert len(atom_groups) == num_res - num_molecules
                dihedrals = get_gamma_sets(atom_groups, dihedral_type)
            all_dihedrals += dihedrals
    else:
        all_dihedrals = None
    edges_and_orders = get_edges_and_orders(
        md_topo,
        bond_edges,
        angle_edges,
        pseudobonds,
        all_dihedrals,
        non_bonded_edges,
        termini_stats=data_generation_dict["termini_stats"],
    )

    coords = np.load(
        data_generation_dict["base_save_dir"]
        + mol_dictionary["tag"]
        + "_cg_coords.npy"
    )[::stride]
    types = np.load(
        data_generation_dict["base_save_dir"]
        + mol_dictionary["tag"]
        + "_cg_embeds.npy"
    )
    sub_data_list, prior_nls = get_atomic_data_list(
        coords, types, edges_and_orders
    )
    return sub_data_list, prior_nls


def multi_chunker(
    atomic_data_list: List[AtomicData],
    full_prior_model: torch.nn.Module,
    num_atoms: int,
    batch_size: int = 1000,
) -> np.array:
    """Takes chunks and pushes them through the prior
    model

    Parameters
    ----------
    atomic_data:
       list of atomic data instances
    full_prior_model:
       full mlcg prior model
    num_atoms:
       Number of CG atoms in the molecule
    batch_size:
       Size of chunks/batches

    Returns
    -------
    delta_forces:
       CG delta forces
    """

    dataloader = DataLoader(
        atomic_data_list, batch_size=batch_size, shuffle=False
    )

    all_delta_forces = []

    for num, data in tqdm(enumerate(dataloader), desc="chunking data..."):
        data = full_prior_model(data)
        delta_forces = data["forces"] - data["out"]["forces"]
        all_delta_forces.append(
            delta_forces.detach().numpy().reshape(batch_size, num_atoms, 3)
        )
    return np.concatenate(all_delta_forces, axis=0)
