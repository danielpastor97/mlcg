import os
import h5py
import numpy as np

from pdb_str_util import (
    read_from_pdb,
    load_mdtraj_from_str,
)  # for handling pdb files in hdf5 files. Not yet used

ROOTDIR = "/import/a12/users/nickc/mlcg_delta_datasets/1_4_res_exclusion/"
TRAIN_DIR = ROOTDIR + "mlcg_train/"
VAL_DIR = ROOTDIR + "mlcg_val/"
CATH_GLOB_MUSTER = "cath_*_cg_coords.npy"
CATH_MUSTER = "cath_%s_cg_coords.npy"
CATH_MUSTER_FORCE = "cath_%s_1_4_res_exclusion_delta_forces_shaped.npy"
CATH_MUSTER_EMBED = "cath_%s_cg_embeds.npy"
OPEP_MUSTER = "opep_%s_cg_coords.npy"
OPEP_MUSTER_FORCE = "opep_%s_1_4_res_exclusion_delta_forces_shaped.npy"
OPEP_MUSTER_EMBED = "opep_%s_cg_embeds.npy"


OUTPUT_DIR = "./datasets/"
if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

from glob import glob

# ---- find CATH data files and accumulate them into a HDF5 record ----
cath_names = []
for f in glob(TRAIN_DIR + CATH_GLOB_MUSTER):
    name = f.split("/")[-1][5:12]
    cath_names.append(name)
for f in glob(VAL_DIR + CATH_GLOB_MUSTER):
    name = f.split("/")[-1][5:12]
    cath_names.append(name)
cath_names = sorted(cath_names)
print(cath_names)
print(len(cath_names))


def load_CATH(serial):
    from os import path

    output = {}
    if path.exists(TRAIN_DIR + CATH_MUSTER % serial):
        output["cg_coords"] = np.load(TRAIN_DIR + CATH_MUSTER % serial)
        output["cg_delta_forces"] = np.load(
            TRAIN_DIR + CATH_MUSTER_FORCE % serial
        )
        output["cg_embeds"] = np.load(TRAIN_DIR + CATH_MUSTER_EMBED % serial)
    else:
        output["cg_coords"] = np.load(VAL_DIR + CATH_MUSTER % serial)
        output["cg_delta_forces"] = np.load(
            VAL_DIR + CATH_MUSTER_FORCE % serial
        )
        output["cg_embeds"] = np.load(VAL_DIR + CATH_MUSTER_EMBED % serial)
    return output


f = h5py.File(OUTPUT_DIR + "cath_1_4_res_exclusion.h5", "w")
metaset = f.create_group("CATH")
for c_name in cath_names:
    name = "cath_%s" % c_name
    hdf_group = metaset.create_group(name)
    cath_data = load_CATH(c_name)
    hdf_group.create_dataset("cg_coords", data=cath_data["cg_coords"])
    hdf_group.create_dataset(
        "cg_delta_forces", data=cath_data["cg_delta_forces"]
    )
    hdf_group.attrs["cg_embeds"] = cath_data["cg_embeds"]
    hdf_group.attrs["N_frames"] = cath_data["cg_coords"].shape[0]
    print("Processed cath_%s" % c_name)
f.close()

# ---- find OPEP data files and accumulate them into a HDF5 record ----


def load_OPEP(serial):
    from os import path

    output = {}
    serial = "%04d" % serial
    if path.exists(TRAIN_DIR + OPEP_MUSTER % serial):
        output["cg_coords"] = np.load(TRAIN_DIR + OPEP_MUSTER % serial)
        output["cg_delta_forces"] = np.load(
            TRAIN_DIR + OPEP_MUSTER_FORCE % serial
        )
        output["cg_embeds"] = np.load(TRAIN_DIR + OPEP_MUSTER_EMBED % serial)
    else:
        output["cg_coords"] = np.load(VAL_DIR + OPEP_MUSTER % serial)
        output["cg_delta_forces"] = np.load(
            VAL_DIR + OPEP_MUSTER_FORCE % serial
        )
        output["cg_embeds"] = np.load(VAL_DIR + OPEP_MUSTER_EMBED % serial)
    return output


f = h5py.File(OUTPUT_DIR + "opep_1_4_res_exclusion.h5", "w")
metaset = f.create_group("OPEP")
for i in range(1100):
    name = "opep_%04d" % i
    hdf_group = metaset.create_group(name)
    opep_data = load_OPEP(i)
    hdf_group.create_dataset("cg_coords", data=opep_data["cg_coords"])
    hdf_group.create_dataset(
        "cg_delta_forces", data=opep_data["cg_delta_forces"]
    )
    hdf_group.attrs["cg_embeds"] = opep_data["cg_embeds"]
    hdf_group.attrs["N_frames"] = opep_data["cg_coords"].shape[0]
    print("Processed opep_%04d" % i)
f.close()

# ---- establish a HDF5 record to merge two Metasets together ----
f = h5py.File(OUTPUT_DIR + "combined_1_4_res_exclusion.h5", "w")
# note: h5py treats the external link as relative path from directory of the main h5py file.
# therefore we don't include `OUTPUT_DIR` below
# and the generated combined file should stay in the same folder as the other two files.
f["OPEP"] = h5py.ExternalLink("opep_1_4_res_exclusion.h5", "/OPEP")
f["CATH"] = h5py.ExternalLink("cath_1_4_res_exclusion.h5", "/CATH")
f.close()
