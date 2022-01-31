import os.path as osp
import h5py
import numpy as np
import argparse
from glob import glob
from tqdm import tqdm
from torch_geometric.data.makedirs import makedirs


from pdb_str_util import (
    read_from_pdb,
    load_mdtraj_from_str,
)  # for handling pdb files in hdf5 files. Not yet used

# define some
CATH_GLOB_MUSTER = "cath_*_cg_coords.npy"
CATH_MUSTER = "cath_%s_cg_coords.npy"
CATH_MUSTER_FORCE = "cath_%s_{outname}_delta_forces_shaped.npy"
CATH_MUSTER_EMBED = "cath_%s_cg_embeds.npy"
OPEP_MUSTER = "opep_%s_cg_coords.npy"
OPEP_MUSTER_FORCE = "opep_%s_{outname}_delta_forces_shaped.npy"
OPEP_MUSTER_EMBED = "opep_%s_cg_embeds.npy"


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rd",
        "--rootdir",
        metavar="FN",
        default="/import/a12/users/nickc/mlcg_delta_datasets/1_4_res_exclusion/",
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

    return parser.parse_args()


def load_CATH(serial, outname):
    output = {}
    force_fn = CATH_MUSTER_FORCE.format(outname=outname)
    fn = osp.join(TRAIN_DIR, CATH_MUSTER % serial)
    if osp.exists(fn):
        output["cg_coords"] = np.load(fn)
        output["cg_delta_forces"] = np.load(
            osp.join(TRAIN_DIR, force_fn % serial)
        )
        output["cg_embeds"] = np.load(
            osp.join(TRAIN_DIR, CATH_MUSTER_EMBED % serial)
        )
    else:
        output["cg_coords"] = np.load(osp.join(VAL_DIR, CATH_MUSTER % serial))
        output["cg_delta_forces"] = np.load(
            osp.join(VAL_DIR, force_fn % serial)
        )
        output["cg_embeds"] = np.load(
            osp.join(VAL_DIR, CATH_MUSTER_EMBED % serial)
        )
    return output


def load_OPEP(serial, outname):
    output = {}
    serial = "%04d" % serial
    force_fn = OPEP_MUSTER_FORCE.format(outname=outname)
    if osp.exists(osp.join(TRAIN_DIR, OPEP_MUSTER % serial)):
        output["cg_coords"] = np.load(osp.join(TRAIN_DIR, OPEP_MUSTER % serial))
        output["cg_delta_forces"] = np.load(
            osp.join(TRAIN_DIR, force_fn % serial)
        )
        output["cg_embeds"] = np.load(
            osp.join(TRAIN_DIR, OPEP_MUSTER_EMBED % serial)
        )
    else:
        output["cg_coords"] = np.load(osp.join(VAL_DIR, OPEP_MUSTER % serial))
        output["cg_delta_forces"] = np.load(
            osp.join(VAL_DIR, force_fn % serial)
        )
        output["cg_embeds"] = np.load(
            osp.join(VAL_DIR, OPEP_MUSTER_EMBED % serial)
        )
    return output


if __name__ == "__main__":
    args = parse_cli()

    OUTPUT_DIR = args.outdir
    makedirs(OUTPUT_DIR)

    ROOTDIR = args.rootdir
    TRAIN_DIR = osp.join(ROOTDIR, "mlcg_train")
    VAL_DIR = osp.join(ROOTDIR, "mlcg_val")

    outname = osp.basename(osp.abspath(ROOTDIR))

    # ---- find CATH data files and accumulate them into a HDF5 record ----
    cath_names = []
    for f in glob(osp.join(TRAIN_DIR, CATH_GLOB_MUSTER)):
        name = f.split("/")[-1][5:12]
        cath_names.append(name)
    for f in glob(osp.join(VAL_DIR, CATH_GLOB_MUSTER)):
        name = f.split("/")[-1][5:12]
        cath_names.append(name)
    cath_names = sorted(cath_names)
    print("Found these CATH protein:")
    print(cath_names)
    print(len(cath_names))

    with h5py.File(osp.join(OUTPUT_DIR, f"cath_{outname}.h5"), "w") as f:
        metaset = f.create_group("CATH")
        for c_name in tqdm(cath_names, desc="process CATH"):
            name = "cath_%s" % c_name
            hdf_group = metaset.create_group(name)
            cath_data = load_CATH(c_name, outname)
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
            opep_data = load_OPEP(i, outname)
            hdf_group.create_dataset("cg_coords", data=opep_data["cg_coords"])
            hdf_group.create_dataset(
                "cg_delta_forces", data=opep_data["cg_delta_forces"]
            )
            hdf_group.attrs["cg_embeds"] = opep_data["cg_embeds"]
            hdf_group.attrs["N_frames"] = opep_data["cg_coords"].shape[0]
            # print("Processed opep_%04d" % i)

    # ---- establish a HDF5 record to merge two Metasets together ----
    with h5py.File(osp.join(OUTPUT_DIR, f"combined_{outname}.h5"), "w") as f:
        # note: h5py treats the external link as relative path from directory of the main h5py file.
        # therefore we don't include `OUTPUT_DIR` below
        # and the generated combined file should stay in the same folder as the other two files.
        f["OPEP"] = h5py.ExternalLink(f"opep_{outname}.h5", "/OPEP")
        f["CATH"] = h5py.ExternalLink(f"cath_{outname}.h5", "/CATH")
