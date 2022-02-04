import numpy, h5py
from mlcg.datasets import multimol_split, n_fold_multi_mol_split
import argparse
from ruamel.yaml import YAML


def prepare_molecule_dictionary(h5_file):
    f = h5py.File(h5_file)

    mol_dict = {}
    subset_names = list[f.keys()]
    for name in subset_names:
        for k in f[name]:
            mol_dict[k] = f["name"][k].attrs["N_frames"]
    return mol_dict


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-h5",
        "--h5path",
        type=str,
        default="/import/a12/users/nickc/mlcg_delta_datasets/dihedral_1_6_res_exclusion/combined_dihedral_1_6_res_exclusion.h5",
        help="path to the h5 dataset",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default="./partions.yaml",
        help="filename where the splits will be saved in YAML format",
    )
    parser.add_argument(
        "--kfsplits",
        default=None,
        type=int,
        help="If specified, the number of folds to split the dataset into. Otherwise, a single split will be made.",
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="If specified, the splitting or KFfold splitting will be seeded",
    )
    parser.add_argument(
        "--proportions",
        nargs="+",
        default=[0.8, 0.2],
        type=list,
        help="For non-KFold splitting, this list gives [train_proportion, validation_proportion]",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli()
    yaml = YAML()

    mol_dict = prepare_molecule_dictionary(args.h5path)

    if args.kfsplits != None:
        k_fold_splits = n_fold_multi_mol_split(
            mol_dict, k=args.kfsplits, shuffle=True, random_state=seed
        )
        with open(args.outfile, "wb") as yfile:
            yaml.dump(k_fold_splits, yfile)

    else:
        splits = multimol_split(
            mol_dict,
            proportions=args.proportions,
            random_seed=args.seed,
            verbose=verbose,
        )
        with open(args.outfile, "wb") as yfile:
            yaml.dump(splits, yfile)
