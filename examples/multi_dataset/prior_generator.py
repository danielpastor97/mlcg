from multi_data_tools import *
from generator_routines import *
from tqdm import tqdm
import yaml
import sys
import argparse


def main():
    args = sys.argv[1:]
    options, parser = argparser(args)
    data_gen_opts = yaml.safe_load(open(options.config, "rb"))

    if options.save:
        save_cg_data(data_gen_opts)
    if options.fit:
        if data_gen_opts["load_accumulation"] != None:
            print("loading pre-collated data...")
            collated_data = torch.load(data_gen_opts["load_accumulation"])
        else:
            print("collating data...")
            collated_data = accumulate_data(data_gen_opts)
        if data_gen_opts["save_accumulation"] != None:
            print("Saving collated data...")
            torch.save(collated_data, data_gen_opts["save_accumulation"])
        if not options.nls_only:
            prior_model, all_stats = fit_transferable_baseline_model(
                collated_data, data_gen_opts
            )
    if options.produce:
        save_delta_forces(data_gen_opts)
    if options.targets:
        generate_targets(data_gen_opts)


def argparser(args):
    parser = argparse.ArgumentParser(
        description="Script for generating CG data and prior models for transferable datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "config", help="configuration file for CG data generation"
    )
    parser.add_argument(
        "--save", help="save CG data from all-atom data", action="store_true"
    )
    parser.add_argument(
        "--fit", help="fit saved CG data to prior model.", action="store_true"
    )
    parser.add_argument(
        "--produce", help="produce delta forces", action="store_true"
    )
    parser.add_argument(
        "--nls-only",
        help="produce only neighborlists if using --fit without fitting/saving a prior",
        action="store_true",
    )
    parser.add_argument(
        "--targets",
        help="produce coordinates/forces/nls for targets",
        action="store_true",
    )

    return parser.parse_args(args), parser


if __name__ == "__main__":
    main()
