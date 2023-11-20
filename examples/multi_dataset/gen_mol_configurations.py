import sys
import numpy as np
import argparse
import torch
import mdtraj as md
from mlcg.data import AtomicData
from _embeddings import generate_embeddings, opep_embedding_map
from multi_data_tools import *
from typing import Sequence, List, Tuple, Union, Dict
import yaml
from copy import deepcopy


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Tool for generating AtomicData lists for starting simulation configurations. OPEP CG mapping is assumed. Outputs a torch-pickled list of AtomicData configurations as well as a prior model specific to the molecular neighborlist. Be aware that CA-CB order is strictly required for all PRO residues. Currently this script is limited for use with a SINGLE topology (though users can input several distant conformations via PDB files). WARNING: all inmput CG PDBs must follow the OPEPS mapping order for every residue - [N,CA,CB,C,O] for non-GLY, [N,CA,C,O] for GLY."
    )

    parser.add_argument(
        "--pdbs", nargs="+", type=str, help="path to single frame PDB files"
    )
    parser.add_argument(
        "--copies",
        type=int,
        help="number of copies to tile for each specified PDB",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="my_configs",
        help="path basename with which output .pt files are saved",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="path to YAML file with dataset/prior options",
    )

    return parser


def generate_neighborlist(
    cg_traj: md.Trajectory, mol_dictionary, data_generation_dict
) -> Dict[str, torch.Tensor]:
    """Processes data accumulation for CG data before collation.
    This is the main loop associated with aggregating molecular
    data just before fitting any priors.

    Parameters
    ----------
    cg_traj:
        CG MDTraj Trajectory
    mol_dictionary:
        CG molecule dictionary
    data_generation_dict:
        data generation dictionary

    Returns
    -------
    prior_nls:
        neighborlist according to data generation options
    """
    types = mol_dictionary["types"]
    res_idx = mol_dictionary["res_idx"]

    traj = cg_traj
    assert all(
        [
            res1.name == res2.name
            for res1, res2 in zip(
                mol_dictionary["residues"], list(traj.topology.residues)
            )
        ]
    )
    assert len(mol_dictionary["cg_atoms"]) == len(list(traj.topology.atoms))

    md_topo = traj.topology

    # Make MLCG topology
    residue_list = list([res.name for res in md_topo.residues])
    num_res = len(residue_list)
    num_gly = residue_list.count("GLY")
    mlcg_topo, bond_edges, angle_edges = get_mlcg_topo(md_topo, types)
    conn_mat = get_connectivity_matrix(mlcg_topo)

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
                assert len(atom_groups) == num_res - num_gly
                dihedrals = get_gamma_sets(atom_groups, dihedral_type)
            if dihedral_type == "gamma_2":
                atom_groups = get_improper_atoms(
                    conn_mat, np.array(types), "gamma_2"
                )
                assert len(atom_groups) == num_res - 1
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

    coords = traj.xyz * 10.0  # convert to angstrom
    types = mol_dictionary["types"]
    _, prior_nls = get_atomic_data_list(coords, types, edges_and_orders)
    return prior_nls


def get_masses(types: Sequence[int], mass_scale=418.4) -> np.ndarray:
    """Generates CG masses according to standard OPEP mapping

    Parameters
    ----------
    types:
        list/array of CG types
    mass_scale:
        Optional mass rescaling: mass = mass / mass_scale

    Returns
    -------
    masses:
        (rescaled) CG masses
    """

    carbons = np.arange(1, 21)  # CB atoms
    N, CA, C, O = 14.0, 12.0, 12.0, 16.0  # backbone atoms
    masses = []
    for t in types:
        if t in carbons:
            masses.append(12.0)
        elif t == 21:
            masses.append(N)
        elif t == 22:
            masses.append(CA)
        elif t == 23:
            masses.append(C)
        elif t == 24:
            masses.append(O)
        else:
            raise ValueError("type {} not in OPEP embedding map".format(t))
    masses = np.array(masses) / mass_scale
    return masses


if __name__ == "__main__":
    parser = parse_cli()
    args = parser.parse_args()
    pdbs = args.pdbs
    data_gen_opts = yaml.safe_load(open(args.config, "rb"))

    prior_path = "{}/full_prior_model_{}.pt".format(
        data_gen_opts["base_save_dir"], data_gen_opts["prior_tag"]
    )
    base_prior_model = torch.load(prior_path).models

    cg_coord_list = []
    cg_type_list = []
    cg_mass_list = []
    cg_traj_list = []
    cg_nls_list = []
    for pdb_file in pdbs:
        pdb = md.load(pdb_file)
        if pdb.n_frames != 1:
            raise ValueError(
                "supplied pdb should have exactly 1 frame, but it instead has {}.".format(
                    pdb.n_frames
                )
            )
        for _ in range(args.copies):
            mol_dictionary = get_mol_dictionary(pdb, pro_swap=False)
            cg_pdb = pdb.atom_slice(mol_dictionary["cg_indices"])
            cg_coords = cg_pdb.xyz * 10.0  # convert from nm to angstroms
            cg_types = mol_dictionary["types"]
            cg_masses = get_masses(cg_types)

            # generate neighborlist and a fresh prior model copy
            nls = generate_neighborlist(cg_pdb, mol_dictionary, data_gen_opts)
            prior_model = deepcopy(base_prior_model)

            assert set(nls.keys()) == set(prior_model.keys())

            cg_coord_list.append(cg_coords)
            cg_type_list.append(cg_types)
            cg_mass_list.append(cg_masses)
            cg_nls_list.append(nls)

    assert (
        len(cg_coord_list)
        == len(cg_type_list)
        == len(cg_mass_list)
        == len(cg_nls_list)
    )
    # build initial configuration list
    data_list = []
    for coords, types, masses, nls in zip(
        cg_coord_list, cg_type_list, cg_mass_list, cg_nls_list
    ):
        data = AtomicData.from_points(
            pos=torch.tensor(coords[0]),
            atom_types=torch.tensor(types),
            masses=torch.tensor(masses),
        )
        data.neighbor_list = deepcopy(nls)
        data_list.append(data)

    torch.save(data_list, args.out + "_configurations.pt")
    torch.save(prior_model, args.out + "_specific_prior.pt")
