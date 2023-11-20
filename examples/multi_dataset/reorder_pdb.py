from typing import List, Tuple, Dict
import mdtraj as md
import numpy as np
from cg_molecule import CGMolecule
import argparse


def opep_sorting(
    traj: md.Trajectory,
) -> Tuple[np.ndarray, List[str], List[int], Dict[int, str]]:
    """Function for taking a CG PDB structure with any ordering
    within of atoms in any residue and reorders them to all
    be the standard OPEP ordering: [N, CA, CB, C, O] for non-GLY
    and [N, CA, C, O] for GLY residues. Input PDBs must be
    single-frame CG PDBs already at the OPEP resolution.

    Parameters
    ----------
    traj:
        Input MDTraj Trajectory, with a single frame, and whose
        topology only includes OPEP atoms at the OPEP resolution

    Returns
    -------
    new_coords:
        The resorted coordinates
    sorted_names:
        The resorted atom name strings
    sorted_resseq:
        The residue ID for each CG atom
    sorted_chainseq:
        The chain ID for each CG atom
    sorted_resmap:
        The map from residue sequence to residue name
    """

    if traj.xyz.shape[0] != 1:
        raise RuntimeError(
            "Supplied PDB has {} frames, but sorting should only be done for single frame PDBs (for now).".format(
                traj.xyz.shape[0]
            )
        )

    acceptable_names = ["N", "CA", "CB", "C", "O"]
    if any([atom.name not in acceptable_names for atom in traj.topology.atoms]):
        raise RuntimeError(
            "There is an atom in the input CG PDB that is not compatible with the OPEP CG mapping"
        )

    original_coords = traj.xyz[0]
    desired_order_non_GLY = ["N", "CA", "CB", "C", "O"]
    desired_order_GLY = ["N", "CA", "C", "O"]
    atom_idx = 0
    new_coords = []
    sorted_names = []
    sorted_resseq = []
    sorted_chainseq = []
    sorted_resmap = {}
    zero_atom_idx = list(traj.topology.atoms)[0].serial
    zero_res_idx = list(traj.topology.residues)[0].resSeq
    zero_chain_idx = list(traj.topology.chains)[0].index
    for chain_id, chain in enumerate(traj.topology.chains):
        for resnum, residue in enumerate(chain.residues):
            original_sub_atom_idx = []
            for atom in list(residue.atoms):
                original_sub_atom_idx.append(atom_idx)
                atom_idx += 1
            original_sub_coords = original_coords[original_sub_atom_idx]
            original_order = [atom.name for atom in list(residue.atoms)]
            indexmap = {k: v for v, k in enumerate(original_order)}
            if residue.name == "GLY":
                re_idx = [indexmap[k] for k in desired_order_GLY]
                if len(original_sub_atom_idx) != 4:
                    raise RuntimeError(
                        "Residue {} does not have the number of CG atoms appropriate for the OPEPs CG mapping.".format(
                            str(residue)
                        )
                    )
            else:
                re_idx = [indexmap[k] for k in desired_order_non_GLY]
                if len(original_sub_atom_idx) != 5:
                    raise RuntimeError(
                        "Residue {} does not have the number of CG atoms appropriate for the OPEPs CG mapping.".format(
                            str(residue)
                        )
                    )

            sorted_sub_coords = original_sub_coords[re_idx]
            new_coords.append(sorted_sub_coords)
            sorted_sub_names = [
                atom.name for atom in np.array(list(residue.atoms))[re_idx]
            ]
            sorted_names.extend(sorted_sub_names)
            sorted_resseq.extend(
                4 * [resnum + zero_res_idx]
                if residue.name == "GLY"
                else 5 * [resnum + zero_res_idx]
            )
            sorted_chainseq.extend(
                4 * [chain_id + zero_chain_idx]
                if residue.name == "GLY"
                else 5 * [chain_id + zero_chain_idx]
            )
            sorted_resmap[resnum + zero_res_idx] = residue.name
    new_coords = np.concatenate(new_coords)
    assert new_coords.shape == original_coords.shape
    return (
        new_coords[None, :],
        sorted_names,
        sorted_resseq,
        sorted_chainseq,
        sorted_resmap,
    )


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Tool that takes a CG PDB at the OPEPs resolution that does not conform to the standard OPEPS per-residue atom ordering and ouputs a PDB for the same system with the correct ordering."
    )

    parser.add_argument(
        "pdb", type=str, help="path to the input, unordered CG PDB"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./my_ordered_cg_pdb.pdb",
        help="path at which the ordered CG PDB is saved",
    )
    return parser


if __name__ == "__main__":
    parser = parse_cli()
    args = parser.parse_args()
    unordered_traj = md.load(args.pdb)
    coords, names, resseq, chainseq, resmap = opep_sorting(unordered_traj)
    ordered_cg_mol = CGMolecule(names, resseq, resmap, chainseq)
    ordered_cg_traj = ordered_cg_mol.make_trajectory(coords)
    ordered_cg_traj.save_pdb(args.out)
