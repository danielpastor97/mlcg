import numpy as np
import os
from tqdm import tqdm
import yaml
import sys
import argparse
import shutil


def copy_routine(
    name: str, prior_tag: str, source_dir: str, dest_dir: str
) -> None:
    """Helper function to copy CG files into training and testing
    directories

    Parameters
    ----------
    name:
        the name of the molecule
    prior_tag:
        the string specified in the data_gen_opts dictionary that
        specfies which delta forces to use
    source_dir:
        the directory from which files will be copied
    dest_dir:
        the directory to which files will be copied
    """
    shutil.copyfile(
        source_dir + "{}_cg_coords.npy".format(name),
        dest_dir + "{}_cg_coords.npy".format(name),
    )
    shutil.copyfile(
        source_dir + "{}_cg_embeds.npy".format(name),
        dest_dir + "{}_cg_embeds.npy".format(name),
    )
    shutil.copyfile(
        source_dir + "{}_{}_delta_forces.npy".format(name, prior_tag),
        dest_dir + "{}_{}_delta_forces.npy".format(name, prior_tag),
    )


def main():
    args = sys.argv[1:]
    options, parser = argparser(args)
    data_gen_opts = yaml.safe_load(open(options.config, "rb"))

    if not os.path.exists(data_gen_opts["train_dir"]):
        os.mkdir(data_gen_opts["train_dir"])
    if not os.path.exists(data_gen_opts["test_dir"]):
        os.mkdir(data_gen_opts["test_dir"])

    if "OPEP" in data_gen_opts["sub_datasets"]:
        train_peptides = np.load(
            data_gen_opts["base_save_dir"]
            + "train_OPEP_"
            + data_gen_opts["prior_tag"]
            + ".npy"
        )
        test_peptides = np.load(
            data_gen_opts["base_save_dir"]
            + "test_OPEP_"
            + data_gen_opts["prior_tag"]
            + ".npy"
        )
        for num in tqdm(train_peptides, desc="Creating OPEP train data..."):
            name = "{}{:04d}".format(
                data_gen_opts["sub_datasets"]["OPEP"]["base_tag"], num
            )
            copy_routine(
                name,
                data_gen_opts["prior_tag"],
                data_gen_opts["base_save_dir"],
                data_gen_opts["train_dir"],
            )
        for num in tqdm(test_peptides, desc="Creating OPEP test data..."):
            name = "{}{:04d}".format(
                data_gen_opts["sub_datasets"]["OPEP"]["base_tag"], num
            )
            copy_routine(
                name,
                data_gen_opts["prior_tag"],
                data_gen_opts["base_save_dir"],
                data_gen_opts["test_dir"],
            )

    if "CATH" in data_gen_opts["sub_datasets"]:
        train_cath_domains = np.load(
            data_gen_opts["base_save_dir"]
            + "train_CATH_"
            + data_gen_opts["prior_tag"]
            + ".npy"
        )
        test_cath_domains = np.load(
            data_gen_opts["base_save_dir"]
            + "test_CATH_"
            + data_gen_opts["prior_tag"]
            + ".npy"
        )
        for domain in tqdm(
            train_cath_domains, desc="Creating CATH train data..."
        ):
            name = "cath_{}".format(domain)
            copy_routine(
                name,
                data_gen_opts["prior_tag"],
                data_gen_opts["base_save_dir"],
                data_gen_opts["train_dir"],
            )
        for domain in tqdm(
            test_cath_domains, desc="Creating CATH test data..."
        ):
            name = "cath_{}".format(domain)
            copy_routine(
                name,
                data_gen_opts["prior_tag"],
                data_gen_opts["base_save_dir"],
                data_gen_opts["test_dir"],
            )

    if "AGG" in data_gen_opts["sub_datasets"]:
        train_cath_domains = np.load(
            data_gen_opts["base_save_dir"]
            + "train_AGG_"
            + data_gen_opts["prior_tag"]
            + ".npy"
        )
        test_cath_domains = np.load(
            data_gen_opts["base_save_dir"]
            + "test_AGG_"
            + data_gen_opts["prior_tag"]
            + ".npy"
        )
        for domain in tqdm(
            train_cath_domains, desc="Creating AGG train data..."
        ):
            name = "aggregate_{}".format(domain)
            copy_routine(
                name,
                data_gen_opts["prior_tag"],
                data_gen_opts["base_save_dir"],
                data_gen_opts["train_dir"],
            )
        for domain in tqdm(test_cath_domains, desc="Creating AGG test data..."):
            name = "aggregate_{}".format(domain)
            copy_routine(
                name,
                data_gen_opts["prior_tag"],
                data_gen_opts["base_save_dir"],
                data_gen_opts["test_dir"],
            )

    if "CATH_UNFOLDED" in data_gen_opts["sub_datasets"]:
        train_cath_domains = np.load(
            data_gen_opts["base_save_dir"]
            + "train_CATH_UNFOLDED_"
            + data_gen_opts["prior_tag"]
            + ".npy"
        )
        test_cath_domains = np.load(
            data_gen_opts["base_save_dir"]
            + "test_CATH_UNFOLDED_"
            + data_gen_opts["prior_tag"]
            + ".npy"
        )
        for domain in tqdm(
            train_cath_domains, desc="Creating CATH train data..."
        ):
            name = "unfolded_{}".format(domain)
            copy_routine(
                name,
                data_gen_opts["prior_tag"],
                data_gen_opts["base_save_dir"],
                data_gen_opts["train_dir"],
            )
        for domain in tqdm(
            test_cath_domains, desc="Creating CATH test data..."
        ):
            name = "unfolded_{}".format(domain)
            copy_routine(
                name,
                data_gen_opts["prior_tag"],
                data_gen_opts["base_save_dir"],
                data_gen_opts["test_dir"],
            )

    if "CATH_UNFOLDED_FINAL" in data_gen_opts["sub_datasets"]:
        train_cath_domains = np.load(
            data_gen_opts["base_save_dir"]
            + "train_CATH_UNFOLDED_FINAL_"
            + data_gen_opts["prior_tag"]
            + ".npy"
        )
        test_cath_domains = np.load(
            data_gen_opts["base_save_dir"]
            + "test_CATH_UNFOLDED_FINAL_"
            + data_gen_opts["prior_tag"]
            + ".npy"
        )
        for domain in tqdm(
            train_cath_domains, desc="Creating CATH train data..."
        ):
            name = "unfolded_final{}".format(domain)
            copy_routine(
                name,
                data_gen_opts["prior_tag"],
                data_gen_opts["base_save_dir"],
                data_gen_opts["train_dir"],
            )
        for domain in tqdm(
            test_cath_domains, desc="Creating CATH test data..."
        ):
            name = "unfolded_final{}".format(domain)
            copy_routine(
                name,
                data_gen_opts["prior_tag"],
                data_gen_opts["base_save_dir"],
                data_gen_opts["test_dir"],
            )

    if "DIMER" in data_gen_opts["sub_datasets"]:
        train_dimer_dimers = np.load(
            data_gen_opts["base_save_dir"]
            + "train_DIMER_"
            + data_gen_opts["prior_tag"]
            + ".npy"
        )
        test_dimer_dimers = np.load(
            data_gen_opts["base_save_dir"]
            + "test_DIMER_"
            + data_gen_opts["prior_tag"]
            + ".npy"
        )
        for dimer in tqdm(
            train_dimer_dimers, desc="Creating DIMER train data..."
        ):
            name = "{}".format(dimer)
            copy_routine(
                name,
                data_gen_opts["prior_tag"],
                data_gen_opts["base_save_dir"],
                data_gen_opts["train_dir"],
            )
        for dimer in tqdm(
            test_dimer_dimers, desc="Creating DIMER test data..."
        ):
            name = "{}".format(dimer)
            copy_routine(
                name,
                data_gen_opts["prior_tag"],
                data_gen_opts["base_save_dir"],
                data_gen_opts["test_dir"],
            )


def argparser(args):
    parser = argparse.ArgumentParser(
        description="Script for generating CG data and prior models for transferable datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "config", help="configuration file for CG data generation"
    )
    return parser.parse_args(args), parser


if __name__ == "__main__":
    main()
