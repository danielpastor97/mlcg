import os.path as osp
import h5py
import numpy as np
import argparse
from glob import glob
from tqdm import tqdm
from torch_geometric.data.makedirs import makedirs
import pickle
from mlcg.utils import load_yaml

from pdb_str_util import (
    read_from_pdb,
    load_mdtraj_from_str,
)  # for handling pdb files in hdf5 files. Not yet used

# define templates for the coordinate, force and embedding files
CATH_GLOB_TEMPL = "cath_*_cg_coords.npy"
CATH_TEMPL = "cath_%s_cg_coords.npy"
CATH_TEMPL_FORCE = "cath_%s_{outname}_delta_forces.npy"
CATH_TEMPL_EMBED = "cath_%s_cg_embeds.npy"
OPEP_TEMPL = "opep_%s_cg_coords.npy"
OPEP_TEMPL_FORCE = "opep_%s_{outname}_delta_forces.npy"
OPEP_TEMPL_EMBED = "opep_%s_cg_embeds.npy"


def parse_cli():
    parser = argparse.ArgumentParser(
        description="""
    Example for converting raw datasets of OPEP and CATH with baselined forces in the form of a collection of `.npy` files into an HDF5 file.
    """
    )
    parser.add_argument(
        "-rd",
        "--rootdir",
        metavar="FN",
        default="/import/a12/users/nickc/mlcg_delta_datasets/dihedral_1_6_res_exclusion/",
        type=str,
        help="path to the raw data including the OPEP and CATH datasets",
    )

    parser.add_argument(
        "-ot",
        "--outdir",
        metavar="FN",
        default="./datasets/",
        type=str,
        help="path for the processed dataset",
    )

    parser.add_argument(
        "--config",
        metavar="fn",
        default="",
        type=str,
        help="prior generator YAML config",
    )

    parser.add_argument(
        "--tag",
        metavar="fn",
        default="",
        type=str,
        help="tag for combined h5 filename",
    )

    return parser.parse_args()


def load_CATH(
    serial,
    outname,
    data_dir="/import/a12/users/nickc/mlcg_base_cg_data_dir_cath_opep_omega_fix_no_double_term_om_force_aggregated_percentile_0.1_correct_hybrid_repul_only_ca_o/",
    prior_tag=None,
):
    output = {}
    force_fn = CATH_TEMPL_FORCE.format(outname=outname)
    fn = osp.join(TRAIN_DIR, CATH_TEMPL % serial)
    if osp.exists(fn):
        output["cg_coords"] = np.load(fn).astype("float32")
        output["cg_delta_forces"] = np.load(
            osp.join(TRAIN_DIR, force_fn % serial)
        ).astype("float32")
        output["cg_embeds"] = np.load(
            osp.join(TRAIN_DIR, CATH_TEMPL_EMBED % serial)
        )
    else:
        output["cg_coords"] = np.load(
            osp.join(VAL_DIR, CATH_TEMPL % serial)
        ).astype("float32")
        output["cg_delta_forces"] = np.load(
            osp.join(VAL_DIR, force_fn % serial)
        ).astype("float32")
        output["cg_embeds"] = np.load(
            osp.join(VAL_DIR, CATH_TEMPL_EMBED % serial)
        )
    return output


def load_OPEP(
    serial,
    outname,
    data_dir="/import/a12/users/nickc/mlcg_base_cg_data_dir_cath_opep_omega_fix_no_double_term_om_force_aggregated_percentile_0.1_correct_hybrid_repul_only_ca_o/",
    prior_tag=None,
):
    output = {}
    serial = "%04d" % serial
    force_fn = OPEP_TEMPL_FORCE.format(outname=outname)
    if osp.exists(osp.join(TRAIN_DIR, OPEP_TEMPL % serial)):
        output["cg_coords"] = np.load(
            osp.join(TRAIN_DIR, OPEP_TEMPL % serial)
        ).astype("float32")
        output["cg_delta_forces"] = np.load(
            osp.join(TRAIN_DIR, force_fn % serial)
        ).astype("float32")
        output["cg_embeds"] = np.load(
            osp.join(TRAIN_DIR, OPEP_TEMPL_EMBED % serial)
        )
    else:
        output["cg_coords"] = np.load(
            osp.join(VAL_DIR, OPEP_TEMPL % serial)
        ).astype("float32")
        output["cg_delta_forces"] = np.load(
            osp.join(VAL_DIR, force_fn % serial)
        ).astype("float32")
        output["cg_embeds"] = np.load(
            osp.join(VAL_DIR, OPEP_TEMPL_EMBED % serial)
        )
    return output


if __name__ == "__main__":
    args = parse_cli()

    OUTPUT_DIR = args.outdir
    makedirs(OUTPUT_DIR)

    ROOTDIR = args.rootdir
    TRAIN_DIR = osp.join(ROOTDIR, "mlcg_train")
    VAL_DIR = osp.join(ROOTDIR, "mlcg_val")
    CONFIG = load_yaml(args.config)
    TAG = args.tag

    outname = osp.basename(osp.abspath(ROOTDIR))
    # ---- find CATH data files and accumulate them into a HDF5 record ----
    cath_names = []
    for f in glob(osp.join(TRAIN_DIR, CATH_GLOB_TEMPL)):
        name = f.split("/")[-1][5:12]
        cath_names.append(name)
    for f in glob(osp.join(VAL_DIR, CATH_GLOB_TEMPL)):
        name = f.split("/")[-1][5:12]
        cath_names.append(name)
    cath_names = sorted(set(cath_names))
    print("Found these CATH protein:")
    print(cath_names)
    print(len(cath_names))

    with h5py.File(osp.join(OUTPUT_DIR, f"cath_{outname}.h5"), "w") as f:
        metaset = f.create_group("CATH")
        for c_name in tqdm(cath_names, desc="process CATH"):
            name = "cath_%s" % c_name
            hdf_group = metaset.create_group(name)

            cath_data = load_CATH(
                c_name, outname, prior_tag=CONFIG["prior_tag"]
            )

            hdf_group.create_dataset("cg_coords", data=cath_data["cg_coords"])
            hdf_group.create_dataset(
                "cg_delta_forces", data=cath_data["cg_delta_forces"]
            )
            hdf_group.attrs["cg_embeds"] = cath_data["cg_embeds"]
            hdf_group.attrs["N_frames"] = cath_data["cg_coords"].shape[0]
            # print("Processed cath_%s" % c_name)

    # ---- find OPEP data files and accumulate them into a HDF5 record ----

    with h5py.File(osp.join(OUTPUT_DIR, f"opep_{outname}.h5"), "w") as f:
        metaset = f.create_group("OPEP")
        for i in tqdm(range(1100), desc="process OPEP"):
            name = "opep_%04d" % i
            hdf_group = metaset.create_group(name)
            opep_data = load_OPEP(i, outname, prior_tag=CONFIG["prior_tag"])
            hdf_group.create_dataset("cg_coords", data=opep_data["cg_coords"])
            hdf_group.create_dataset(
                "cg_delta_forces", data=opep_data["cg_delta_forces"]
            )
            hdf_group.attrs["cg_embeds"] = opep_data["cg_embeds"]
            hdf_group.attrs["N_frames"] = opep_data["cg_coords"].shape[0]
            # print("Processed opep_%04d" % i)

    # ---- establish a HDF5 record to merge two Metasets together ----
    with h5py.File(
        osp.join(OUTPUT_DIR, f"{TAG}_combined_{outname}.h5"), "w"
    ) as f:
        # note: h5py treats the external link as relative path from directory of the main h5py file.
        # therefore we don't include `OUTPUT_DIR` below
        # and the generated combined file should stay in the same folder as the other two files.
        f["OPEP"] = h5py.ExternalLink(f"opep_{outname}.h5", "/OPEP")
        f["CATH"] = h5py.ExternalLink(f"cath_{outname}.h5", "/CATH")
