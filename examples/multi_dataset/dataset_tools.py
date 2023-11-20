from typing import List, Tuple, Dict, Callable, Optional
from tqdm import tqdm
import re
from mdtraj.core.topology import Atom
import numpy as np
import mdtraj as md
import os
from glob import glob
from cgnet.molecule import CGMolecule
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
import h5py
from two_peptides.h5 import string_to_topology


def natural_sort(name: str) -> str:
    """Performs natural sort"""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(name, key=alphanum_key)


### GET FUNCTIONS #############################################################
# These functions are used to grab filenames and pdbs for each dataset        #
###############################################################################


def get_opep(
    num: int, base_dir: str = "/storage/nickc/octapeptides/"
) -> Tuple[Trajectory, List[str], List[str]]:
    """Takes a peptide number from 0 to 1100 and returns a
    loaded MDTraj object according to its pdb

    Returns
    -------
    pdb:
        MDTraj trajectory instance of specified octapeptide
    coord_files:
        list of all-atom coordinate files associated with the supplied peptide number
    force_files:
        list of all-atom force files associated with the supplied peptide number
    """
    if num < 200:
        coord_files = sorted(
            glob(base_dir + "coords_nowater/opep_{:04d}/".format(num) + "*.npy")
        )
        force_files = sorted(
            glob(base_dir + "forces_nowater/opep_{:04d}/".format(num) + "*.npy")
        )
        assert len(coord_files) == len(force_files)
        pdb_file = (
            base_dir
            + "largepeptides/opep_{:04d}/filtered/filtered.pdb".format(num)
        )
        pdb = md.load_pdb(pdb_file)
    else:  # structures supplied by Adria
        coord_files = sorted(
            glob(
                base_dir
                + "coords_nowater/coor_opep_{:04d}_".format(num)
                + "*.npy"
            )
        )
        force_files = sorted(
            glob(
                base_dir
                + "forces_nowater/force_opep_{:04d}_".format(num)
                + "*.npy"
            )
        )
        assert len(coord_files) == len(force_files)
        pdb_file = (
            base_dir
            + "largepeptides_2/opep_{:04d}/input/e1s1_opep_{:04d}/structure.pdb".format(
                num, num
            )
        )
        pdb = md.load_pdb(pdb_file)
    filenames = [
        (coord_file, force_file)
        for coord_file, force_file in zip(coord_files, force_files)
    ]
    return pdb, filenames


def get_constraint_projected_opep(
    num: int,
    base_dir: str = "/storage/nickc/octapeptides/",
    coordforce_base_dir: str = "/import/a12/users/yaoyic/allegro_backup/opep_proj/",
) -> Tuple[Trajectory, List[str], List[str]]:
    """Takes a peptide number from 0 to 1100 and returns a
    loaded MDTraj object according to its pdb.

    NOTE
    ----
    These are constraint projected forces, as processed by `bgmol.utils.constraints`,
    correcting for spurious forces from absent constrained atoms, eg SHAKE-constrained
    hydrogens.

    Returns
    -------
    pdb:
        MDTraj trajectory instance of specified octapeptide
    coord_files:
        list of all-atom coordinate files associated with the supplied peptide number
    force_files:
        list of all-atom force files associated with the supplied peptide number
    """
    if num < 200:
        pdb_file = (
            base_dir
            + "largepeptides/opep_{:04d}/filtered/filtered.pdb".format(num)
        )
        pdb = md.load_pdb(pdb_file)
    else:  # structures supplied by Adria
        pdb_file = (
            base_dir
            + "largepeptides_2/opep_{:04d}/input/e1s1_opep_{:04d}/structure.pdb".format(
                num, num
            )
        )
        pdb = md.load_pdb(pdb_file)
    coord_file = (
        coordforce_base_dir
        + "opep_{0:04d}_all_atom_constraint_projected_coordinates.npy".format(
            num
        )
    )
    force_file = (
        coordforce_base_dir
        + "opep_{0:04d}_all_atom_constraint_projected_forces.npy".format(num)
    )
    filenames = [(coord_file, force_file)]
    return pdb, filenames


def get_CATH(
    domain_name: str, base_dir: str = "/import/a12/users/nickc/updated_cath/"
) -> Tuple[Trajectory, List[str]]:
    """Takes a CATH domain name string and returns
    a loaded MDTraj object according to its pdb, as well as the associated
    all atom coordinate and force files.

    Returns
    -------
    pdb:
        MDTraj trajectory instance of specified CATH domain
    data_dict_fns:
        Numpy dictionary of coordinates and forces for the specified CATH domain
    """
    pdb_file = base_dir + "pdbs/{}_eq.pdb".format(domain_name)
    pdb = md.load_pdb(pdb_file)

    # isolate just the protein
    prot_idx = pdb.top.select("protein")
    protein_top = pdb.top.subset(prot_idx)
    data_dict_fns = natural_sort(
        glob(base_dir + "output/" + domain_name + "/*_part_*")
    )
    return pdb, data_dict_fns


def get_CATH_unfolded(
    domain_name: str, base_dir: str = "/import/a12/users/aguljas/cath_unfolded/"
) -> Tuple[Trajectory, List[str]]:
    """Takes an unfolded CATH domain name string and returns
    a loaded MDTraj object according to its pdb, as well as the associated
    all atom unfolded coordinate and force files.
    Returns
    -------
    pdb:
        MDTraj trajectory instance of specified CATH domain
    data_dict_fns:
        Numpy dictionary of coordinates and forces for the specified CATH domain
    """
    pdb_file = base_dir + domain_name + "/{}_unfolded.pdb".format(domain_name)
    pdb = md.load_pdb(pdb_file)
    coord_files = sorted(glob(base_dir + domain_name + "/coords*npy"))
    force_files = sorted(glob(base_dir + domain_name + "/forces*npy"))
    filenames = [
        (coord_file, force_file)
        for coord_file, force_file in zip(coord_files, force_files)
    ]
    return pdb, filenames


def get_aggregate(
    structure: str,
    base_dir: str = "/import/a12/users/aguljas/aggregate_dataset/",
) -> Tuple[Trajectory, List[str]]:
    """Takes a aggregate structure name string and returns
    a loaded MDTraj object according to its pdb, as well as the associated
    all atom unfolded coordinate and force files.
    Returns
    -------
    pdb:
        MDTraj trajectory instance of specified CATH domain
    data_dict_fns:
        Numpy dictionary of coordinates and forces for the specified CATH domain
    """
    name, rep = structure[:-5], structure[-4:]
    pdb_file = base_dir + name + "/{}-2layer.pdb".format(name)
    pdb = md.load(pdb_file)
    filenames = sorted(
        glob(base_dir + name + f"/outputs_2layer/outputs_{rep}.npz")
    )
    return pdb, filenames


def get_CATH_unfolded_final(
    domain_name: str,
    base_dir: str = "/import/a12/users/aguljas/cath_unfolded_final/",
) -> Tuple[Trajectory, List[str]]:
    """Takes an unfolded CATH domain name string and returns
    a loaded MDTraj object according to its pdb, as well as the associated
    all atom unfolded coordinate and force files.
    Returns
    -------
    pdb:
        MDTraj trajectory instance of specified CATH domain
    data_dict_fns:
        Numpy dictionary of coordinates and forces for the specified CATH domain
    """
    pdb_file = base_dir + domain_name + "/{}_fullsys_eq.pdb".format(domain_name)
    pdb = md.load_pdb(pdb_file)

    # isolate just the protein
    prot_idx = pdb.top.select("protein")
    pdb = pdb.atom_slice(prot_idx)
    data_dict_fns = sorted(glob(base_dir + domain_name + "/outputs/*_run1.npz"))
    data_dict_fns += sorted(
        glob(base_dir + domain_name + "/outputs/*_run2.npz")
    )
    return pdb, data_dict_fns


def get_constraint_projected_cath(
    domain_name: str,
    base_dir: str = "/import/a12/users/nickc/updated_cath/",
    coordforce_dir: str = "/import/a12/users/yaoyic/allegro_backup/updated_cath_proj/projected_forces/",
) -> Tuple[Trajectory, List[Tuple[str, str]]]:
    """Takes a CATH domain name string and returns
    a loaded MDTraj object according to its pdb, as well as the associated
    all atom coordinate and force files.

    NOTE: These forces have been constraint projected to account for unrecorded hydrogen-heavy atom bonds
    in reference simulation data.

    Returns
    -------
    pdb:
        MDTraj trajectory instance of specified CATH domain
    files:
        List of coordinate-force filename tuples
    """
    pdb_file = base_dir + "pdbs/{}_eq.pdb".format(domain_name)
    pdb = md.load_pdb(pdb_file)

    # isolate just the protein
    prot_idx = pdb.top.select("protein")
    protein_top = pdb.top.subset(prot_idx)
    coord_files = sorted(
        glob(
            coordforce_dir
            + "cath_{}_all_atom_constraint_projected_coordinates*".format(
                domain_name
            )
        )
    )
    force_files = sorted(
        glob(
            coordforce_dir
            + "cath_{}_all_atom_constraint_projected_forces*".format(
                domain_name
            )
        )
    )
    files = []
    for coord_file, force_file in zip(coord_files, force_files):
        assert coord_file.split("/")[-1] == force_file.split("/")[-1].replace(
            "forces", "coordinates"
        )
        files.append((coord_file, force_file))
    return pdb, files


def get_cln(
    name: str, base_dir: str = "/storage/nickc/octapeptides/"
) -> Tuple[Trajectory, List[str], List[str]]:
    """Takes a CLN trajectory ID and returns a
    loaded MDTraj object according to its pdb

    Returns
    -------
    pdb:
        MDTraj trajectory instance of specified octapeptide
    coord_files:
        list of all-atom coordinate files associated with the supplied peptide number
    force_files:
        list of all-atom force files associated with the supplied peptide number
    """
    coord_files = sorted(glob(base_dir + "coords_nowater/*{}*".format(name)))
    force_files = sorted(glob(base_dir + "forces_nowater/*{}*".format(name)))
    assert len(coord_files) == len(force_files)
    pdb_file = base_dir + "chignolin.pdb"
    pdb = md.load_pdb(pdb_file)
    filenames = [
        (coord_file, force_file)
        for coord_file, force_file in zip(coord_files, force_files)
    ]
    return pdb, filenames


def get_bba(
    name: str, base_dir: str = "/import/a12/users/nickc/bba/"
) -> Tuple[Trajectory, List[str], List[str]]:
    """Takes a BBA trajectory ID and returns a
    loaded MDTraj object according to its pdb.

    NOTE: Not sure how the PDBs are organized at the source,
    so currently the first PDB is used for all trajectories

    Returns
    -------
    pdb:
        MDTraj trajectory instance of specified octapeptide
    coord_files:
        list of all-atom coordinate files associated with the supplied peptide number
    force_files:
        list of all-atom force files associated with the supplied peptide number
    """
    coord_files = sorted(glob(base_dir + "coords_nowater/*{}*".format(name)))
    force_files = sorted(glob(base_dir + "forces_nowater/*{}*".format(name)))
    assert len(coord_files) == len(force_files)
    pdb_file = base_dir + "bba_50ns_0/structure.pdb"
    pdb = md.load_pdb(pdb_file)
    filenames = [
        (coord_file, force_file)
        for coord_file, force_file in zip(coord_files, force_files)
    ]
    return pdb, filenames


def get_villin(
    name: str, base_dir: str = "/import/a12/users/nickc/villin/"
) -> Tuple[Trajectory, List[str], List[str]]:
    """Grabs the coordinates/forces/structure for the desired Villin simulation tag."""
    coord_files = sorted(glob(base_dir + "coords_nowater/*{}*".format(name)))
    force_files = sorted(glob(base_dir + "forces_nowater/*{}*".format(name)))
    # additional filtration

    assert len(coord_files) == len(force_files)
    pdb_file = (
        base_dir + "generators_10/villin_10ns_0/structure.pdb"
    )  # for now, lets use the same PDB file for all
    pdb = md.load_pdb(pdb_file)
    filenames = [
        (coord_file, force_file)
        for coord_file, force_file in zip(coord_files, force_files)
    ]
    return pdb, filenames


def get_wwdomain(
    name: str, base_dir: str = "/import/a12/users/nickc/wwdomain/"
) -> Tuple[Trajectory, List[str], List[str]]:
    """Grabs the coordinates/forces/structure for the desired wwdomain simulation tag."""
    coord_files = sorted(glob(base_dir + "coords_nowater/*{}*".format(name)))
    force_files = sorted(glob(base_dir + "forces_nowater/*{}*".format(name)))
    # additional filtration

    assert len(coord_files) == len(force_files)
    pdb_file = (
        base_dir + "generators/WWdomain_50ns_0/structure.pdb"
    )  # for now, lets use the same PDB file for all
    pdb = md.load_pdb(pdb_file)
    filenames = [
        (coord_file, force_file)
        for coord_file, force_file in zip(coord_files, force_files)
    ]
    return pdb, filenames


def get_lambda_repressor(
    name: str, base_dir: str = "/import/a12/users/nickc/lambda_repressor/"
) -> Tuple[Trajectory, List[str], List[str]]:
    """Grabs the coordinates/forces/structure for the desired lambda repressor simulation tag."""
    coord_files = sorted(glob(base_dir + "coords_nowater/*{}*".format(name)))
    force_files = sorted(glob(base_dir + "forces_nowater/*{}*".format(name)))

    assert len(coord_files) == len(force_files)
    pdb_file = (
        base_dir + "lambda_50ns_0/structure.pdb"
    )  # for now, lets use the same PDB file for all
    pdb = md.load_pdb(pdb_file)
    filenames = [
        (coord_file, force_file)
        for coord_file, force_file in zip(coord_files, force_files)
    ]
    return pdb, filenames


def get_desres_wwdomain(
    name: str,
    base_dir: str = "/import/a12/users/nickc/bgmol_data/wwdomain/",
) -> Tuple[Trajectory, List[str], List[str]]:
    """Grabs the coordinates/structure for DESRES WW-domain. There are no forces."""
    coord_files = sorted(glob(base_dir + "aa_coords/{}.npy".format(name)))
    force_files = [None for _ in range(len(coord_files))]

    pdb_file = (
        base_dir + "wwdomain.pdb"
    )  # for now, lets use the same PDB file for all
    pdb = md.load_pdb(pdb_file).remove_solvent()
    pdb = pdb.atom_slice(pdb.topology.select("protein"))
    filenames = [
        (coord_file, force_file)
        for coord_file, force_file in zip(coord_files, force_files)
    ]
    return pdb, filenames


def get_amber_cln(
    name: str,
    base_dir: str = "/import/a12/users/yaoyic/allegro_backup/chi_amber/",
) -> Tuple[Trajectory, List[str], List[str]]:
    """Takes a CLN trajectory ID and returns a
    loaded MDTraj object according to its pdb

    Returns
    -------
    pdb:
        MDTraj trajectory instance of specified AMBER CLN trajectory
    data_dict_fns:
        Numpy dictionary of coordinates and forces for the specified CATH domain
    """
    data_dict_fns = natural_sort(glob(base_dir + "output/" + name + "_prod_*"))
    pdb_file = base_dir + "system/chi_vac.pdb"
    pdb = md.load_pdb(pdb_file)
    return pdb, data_dict_fns


def get_trpcage(
    name: str, base_dir: str = "/import/a12/users/aguljas/trpcage_amber/"
) -> Tuple[Trajectory, List[str], List[str]]:
    """Takes a TRPcage trajectory ID and returns a
    loaded MDTraj object according to its pdb

    Returns
    -------
    pdb:
        MDTraj trajectory of TRPcage
    coord_files:
        list of all-atom coordinate files
    force_files:
        list of all-atom force files
    """
    coord_files = sorted(glob(base_dir + "coords_raw/trp{}*".format(name)))
    force_files = sorted(glob(base_dir + "forces_raw/trp{}*".format(name)))
    assert len(coord_files) == len(force_files)
    pdb_file = base_dir + "topology_allatom/trp01.pdb"
    pdb = md.load_pdb(pdb_file)
    filenames = [
        (coord_file, force_file)
        for coord_file, force_file in zip(coord_files, force_files)
    ]
    return pdb, filenames


def get_dimer(
    name: str, base_dir="/import/a12/users/kraemea88/two_peptides/data/"
) -> Tuple[Trajectory, List[str]]:
    """Returns a dimer system all-atom solute PDB for a given system name"""
    pdb = md.load(base_dir + "raw/" + name + "_solute.pdb")
    return pdb, [base_dir + "allatom.h5"]


### GET NAME FUNCTIONS ######################################################
# These functions return lists of moleule names for each dataset            #
#############################################################################


def get_dimer_names(
    base_data_dir: str = "/import/a12/users/kraemea88/two_peptides/data/",
) -> List[str]:
    """Returns a list of names of available dimer systems"""
    with h5py.File(base_data_dir + "allatom.h5", "r") as hfile:
        dimer_names = list(hfile["MINI"].keys())
    return dimer_names


def get_AGG_structure_names(
    base_data_dir: str = "/import/a12/users/aguljas/aggregate_dataset/",
) -> List[str]:
    """Helper function to get aggregate structure name list"""
    structure_fns = sorted(glob(base_data_dir + "*_*"))
    structures = [name.split("/")[-1] for name in structure_fns]
    structure_reps = []
    for name in structures:
        rep_list = [name + f"_rep{r}" for r in range(1, 6)]
        structure_reps.extend(rep_list)
    return structure_reps


def get_CATH_domain_names(
    base_data_dir: str = "/net/data02/nickc/cath_short_sims/output/",
) -> List[str]:
    """Helper function to get CATH domain name list"""
    domain_fns = sorted(glob(base_data_dir + "*"))
    for dom_fn in domain_fns:
        dom_name = str(dom_fn)
        dom_name = dom_name.split("/")[-1]
    domains = [name.split("/")[-1] for name in domain_fns]
    return domains


def get_cln_traj_names(
    base_data_dir: str = "/net/data02/nickc/mlcg_cln/raw/",
) -> List[str]:
    """Helper function to get CLN trajectory names"""
    fns = sorted(glob(base_data_dir + "coords_nowater/*"))
    names = []
    for fn in fns:
        name = str(fn)
        name = name.split("/")[-1]
        name = name[10:]
        name = name.split(".")[0]
        names.append(name)
    return names


def get_villin_traj_names(
    base_data_dir: str = "/import/a12/users/nickc/villin/",
) -> List[str]:
    """Helper function to get individual villin trajectory names"""
    fns = sorted(glob(base_data_dir + "coords_nowater/*"))

    # additional filtration
    fns = [fn for fn in fns if "tica_goal_0" in fn]

    names = []
    for fn in fns:
        name = str(fn)
        name = name.split("/")[-1]
        name = name[12:]
        name = name.split(".")[0] + "." + name.split(".")[1]
        names.append(name)
    return names


def get_wwdomain_traj_names(
    base_data_dir: str = "/import/a12/users/nickc/wwdomain/",
) -> List[str]:
    """Helper function to get individual wwdomain trajectory names"""
    fns = sorted(glob(base_data_dir + "coords_nowater/*"))

    # additional filtration
    fns = [fn for fn in fns if "contacts" not in fn]

    names = []
    for fn in fns:
        name = str(fn)
        name = name.split("/")[-1]
        name = name[14:]
        name = name.split(".")[0]
        names.append(name)
    return names


def get_lambda_repressor_traj_names(
    base_data_dir: str = "/import/a12/users/nickc/lambda_repressor/",
) -> List[str]:
    """Helper function to get individual wwdomain trajectory names"""
    fns = sorted(glob(base_data_dir + "coords_nowater/*"))

    names = []
    for fn in fns:
        name = str(fn)
        name = name.split("/")[-1]
        name = name[20:]
        name = name.split(".")[0]
        names.append(name)
    return names


def get_bba_traj_names(
    base_data_dir: str = "/net/data02/nickc/bba/",
) -> List[str]:
    """Helper function to get CLN trajectory names"""
    fns = sorted(glob(base_data_dir + "coords_nowater/*"))
    names = []
    for fn in fns:
        name = str(fn)
        name = name.split("/")[-1]
        name = name[9:]
        name = name.split(".")[0]
        names.append(name)
    return names


def get_desres_wwdomain_traj_names(
    base_data_dir="/import/a12/users/nickc/bgmol_data/wwdomain/",
) -> List[str]:
    names = sorted(glob(base_data_dir + "aa_coords/*.npy"))
    names = [name.split("/")[-1] for name in names]
    names = [name.split(".")[0] for name in names]
    return names


def get_amber_cln_traj_names(
    base_data_dir: str = "/import/a12/users/yaoyic/allegro_backup/chi_amber/",
) -> List[str]:
    """Helper function to get CLN trajectory names"""
    fns = sorted(glob(base_data_dir + "output/*"))
    names = []
    for fn in fns:
        name = str(fn)
        name = name.split("/")[-1]
        name = name.split(".")[0]
        name = name.split("_")[0] + "_" + name.split("_")[1]
        names.append(name)
    return names


def get_trpcage_traj_names(
    base_data_dir: str = "/import/a12/users/aguljas/trpcage_amber/",
) -> List[str]:
    """Helper function to get CLN trajectory names"""
    fns = sorted(glob(base_data_dir + "coords_raw/*"))
    names = []
    for fn in fns:
        name = str(fn)
        name = name.split("/")[-1]
        name = name.split(".")[0]
        name = name[3:]
        splits = name.split("_")
        name = splits[0] + "_" + splits[1]
        names.append(name)
    return names


### LOADER FUNCTIONS #####################################################################
# These functions define specific coord/force loading schemes for each dataset           #
##########################################################################################


def DIMER_loader(filename: str, tag: str = None) -> Tuple[np.array, np.array]:
    # Here filename is actually just a string specifying the h5 file
    # Here we resort to "strong" closing
    hfile = h5py.File(filename, "r")
    coord = hfile["MINI"][tag]["aa_coords"][:]
    force = hfile["MINI"][tag]["aa_forces"][:]

    # Convert to kcal/mol/angstrom and angstrom
    # from kJ/mol/nm and nm

    coord = coord * 10.0
    force = force / 41.84

    hfile.close()
    return coord, force


def AGG_loader(filename: str, tag: str = None) -> Tuple[np.array, np.array]:
    """Helper function to load aggregate all-atom unfolded data"""
    sim_output = np.load(filename)
    coord = sim_output["coords"]
    force = sim_output["forces"]
    force = force / 4.184  # convert from kJ/mol/ang to kcal/mol/ang
    return coord, force


def OPEP_loader(
    filenames: List[str], tag: str = None
) -> Tuple[np.array, np.array]:
    """Helper function to load OPEP all-atom data"""
    coord = np.load(filenames[0], allow_pickle=True)
    force = np.load(filenames[1], allow_pickle=True)
    return coord, force


def CATH_loader(filename: str, tag: str = None) -> Tuple[np.array, np.array]:
    """Helper function to load CATH all-atom data"""
    data_dict = np.load(filename)
    coord = data_dict["coords"]
    coord = 10.0 * coord  # convert nm to angstroms
    force = data_dict["Fs"]
    force = force / 41.84  # convert to from kJ/mol/nm to kcal/mol/ang
    return coord, force


def CATH_unfolded_loader(
    filenames: List[str], tag: str = None
) -> Tuple[np.array, np.array]:
    """Helper function to load CATH all-atom unfolded data"""
    coord = np.load(filenames[0], allow_pickle=True)[
        10:
    ]  # remove high forces at the beginning
    force = np.load(filenames[1], allow_pickle=True)[
        10:
    ]  # remove high forces at the beginning
    force = force / 4.184  # convert from kJ/mol/ang to kcal/mol/ang
    return coord, force


def CATH_unfolded_final_loader(
    filename: str, tag: str = None
) -> Tuple[np.array, np.array]:
    """Helper function to load CATH all-atom unfolded data"""
    data_dict = np.load(filename)
    coord = data_dict["coords"][10:]
    force = data_dict["forces"][10:]
    force = force / 4.184  # convert from kJ/mol/ang to kcal/mol/ang
    return coord, force


def constraint_projected_CATH_loader(
    filenames: List[str], tag: str = None
) -> Tuple[np.array, np.array]:
    """Helper function to load CATH all-atom data that has been constraint-projected"""
    coord = np.load(filenames[0], allow_pickle=True)
    force = np.load(filenames[1], allow_pickle=True)
    coord = 10.0 * coord  # convert nm to angstroms
    force = force / 41.84  # convert to from kJ/mol/nm to kcal/mol/ang
    return coord, force


def CLN_loader(
    filenames: List[str], tag: str = None
) -> Tuple[np.array, np.array]:
    """Helper function to load cln all-atom data"""
    coord = np.load(filenames[0], allow_pickle=True)
    force = np.load(filenames[1], allow_pickle=True)
    return coord, force


def villin_loader(
    filenames: List[str], tag: str = None
) -> Tuple[np.array, np.array]:
    """Helper function to load villin all-atom data"""
    coord = np.load(filenames[0], allow_pickle=True)
    force = np.load(filenames[1], allow_pickle=True)
    return coord, force


def wwdomain_loader(
    filenames: List[str], tag: str = None
) -> Tuple[np.array, np.array]:
    """Helper function to load wwdomain all-atom data"""
    coord = np.load(filenames[0], allow_pickle=True)
    force = np.load(filenames[1], allow_pickle=True)
    return coord, force


def desres_wwdomain_loader(
    filenames: List[str], tag: str = None
) -> Tuple[np.array, np.array]:
    coord = np.load(filenames[0], allow_pickle=True)
    return coord, None


def lambda_repressor_loader(
    filenames: List[str], tag: str = None
) -> Tuple[np.array, np.array]:
    """Helper function to load lambda repressor all-atom data"""
    coord = np.load(filenames[0], allow_pickle=True)
    force = np.load(filenames[1], allow_pickle=True)
    return coord, force


def amber_cln_loader(
    filename: str, tag: str = None
) -> Tuple[np.array, np.array]:
    """Helper function to load AMBER CLN all-atom data"""
    data_dict = np.load(filename)
    coord = data_dict["coords"]
    force = data_dict["Fs"]
    return coord, force


def TRPcage_loader(filenames: List[str], tag: str = None):
    """Helper function to load AMBER TRPcage all-atom-data"""
    coord = np.load(filenames[0], allow_pickle=True)
    force = np.load(filenames[1], allow_pickle=True)
    return coord, force


def BBA_loader(
    filenames: List[str], tag: str = None
) -> Tuple[np.array, np.array]:
    """Helper function to load OPEP all-atom data"""
    coord = np.load(filenames[0], allow_pickle=True)
    force = np.load(filenames[1], allow_pickle=True)
    return coord, force


### MAPPING FUNCTIONS #####################################################################
# These functions define specific coord/force loading schemes for each dataset           #
##########################################################################################


def get_slice_cg_mapping(aa_traj) -> Tuple[np.ndarray, np.ndarray]:
    """Returns an OPEP slice CG mapping. Capped
    termini are handled such that the non-trivial
    CG map entries begin after/before NME/ACE.

    Parmeters
    ---------
    aa_traj:
        All atom MDTraj Trajectory

    Returns
    -------
    global_map:
        Slice CG coordinate mapping
    force_mape:
        Slice CG force mapping
    """
    aa_traj = aa_traj.remove_solvent()
    res_list = [
        res
        for res in aa_traj.topology.residues
        if res.name in opep_embedding_map.keys()
    ]
    num_gly = [
        res.name for res in res_list if res.name in opep_embedding_map.keys()
    ].count("GLY")
    num_non_gly = len(res_list) - num_gly
    num_cg_atoms = 5 * num_non_gly + 4 * num_gly
    num_aa_atoms = len(list(aa_traj.topology.atoms))
    global_map = []
    last_idx = 0
    ace_pad = 0
    nme_pad = 0
    for i, res in enumerate(aa_traj.topology.residues):
        if res.name not in opep_embedding_map.keys():
            # Handle caps - essentially we must shift all non-trivial CG
            # Map entries of the following to the right by the number of ACE/NME/NHE atoms if present
            if res.name == "ACE":
                ace_pad = len(list(res.atoms))
                continue
            elif res.name in ["NME", "NHE"]:
                nme_pad = len(list(res.atoms))
                continue
            else:
                raise RuntimeError(f"Unknown residue: {res}")

        atom_dict = {
            atom.name: idx + last_idx + ace_pad + nme_pad
            for idx, atom in enumerate(res.atoms)
        }
        if res.name == "GLY":
            cg_atom_map = np.zeros((4, num_aa_atoms))
            atoms_needed = ["N", "CA", "C", "O"]
            aa_idx = [atom_dict[atom_name] for atom_name in atoms_needed]
            for cg_idx, idx in enumerate(aa_idx):
                cg_atom_map[cg_idx, idx] = 1
        else:
            cg_atom_map = np.zeros((5, num_aa_atoms))
            atoms_needed = ["N", "CA", "CB", "C", "O"]
            aa_idx = [atom_dict[atom_name] for atom_name in atoms_needed]
            for cg_idx, idx in enumerate(aa_idx):
                cg_atom_map[cg_idx, idx] = 1
        global_map.append(cg_atom_map)
        last_idx = int(max(atom_dict.values()) + 1)
        # reset ACE/NME shifts
        ace_pad = 0
        nme_pad = 0
    global_map = np.concatenate(global_map, axis=0)
    force_map = global_map
    assert all([sum(row) == 1 for row in global_map])
    assert all([row.tolist().count(1) == 1 for row in global_map])
    assert len(global_map) == num_cg_atoms
    return global_map, force_map
