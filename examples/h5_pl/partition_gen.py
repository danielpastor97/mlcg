import numpy, h5py
from mlcg.datasets import multimol_split


f = h5py.File(
    "/import/a12/users/nickc/mlcg_delta_datasets/dihedral_1_6_res_exclusion/combined_dihedral_1_6_res_exclusion.h5"
)

mol_dict = {}
for k in f["CATH"]:
    mol_dict[k] = f["CATH"][k].attrs["N_frames"]

split_dict = multimol_split(
    mol_dict,
    proportions={"train": 0.8, "val": 0.2},
    random_seed=5513,
    verbose=True,
)
print(split_dict)


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rd",
        "--rootdir",
        default="/import/a12/users/nickc/mlcg_delta_datasets/dihedral_1_6_res_exclusion/combined_dihedral_1_6_res_exclusion.h5",
    )
    parser.add_argument("--kfsplits", default=None)
    parser.add_argument("--kfsplits", default=None)
    parser.add_argument("--seed", default=5513)
    parser.add_argument("--proportions", default=[0.8, 0.2])

    return parser.parse_args()
