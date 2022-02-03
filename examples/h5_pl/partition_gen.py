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
