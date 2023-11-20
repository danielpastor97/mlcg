from tqdm import tqdm
import torch
from multi_data_tools import *
from dataset_tools import *
from _embeddings import *
from sklearn.model_selection import train_test_split
import pickle
from torch_geometric.data.collate import collate
from mlcg.geometry.statistics import compute_statistics
from mlcg.data.atomic_data import AtomicData
from mlcg.nn.prior import *
from mlcg.nn.gradients import *
from mlcg.datasets.utils import remove_baseline_forces, chunker


def save_cg_data(data_generation_dict):
    """Generates and saves CG data for the specified datasets

    Parameters
    ----------
    data_generation_dict:
        dictionary of dataset and prior options
    """
    datasets = data_generation_dict["sub_datasets"]
    for dataset, opts in datasets.items():
        if dataset == "OPEP":
            get_fn = get_opep
            loader = OPEP_loader
            molecules = np.arange(1100)
        if dataset == "CATH":
            get_fn = get_CATH
            loader = CATH_loader
            molecules = get_CATH_domain_names()
        if dataset == "CATH_UNFOLDED":
            get_fn = get_CATH_unfolded
            loader = CATH_unfolded_loader
            molecules = get_CATH_domain_names()
        if dataset == "CATH_UNFOLDED_FINAL":
            get_fn = get_CATH_unfolded_final
            loader = CATH_unfolded_final_loader
            molecules = get_CATH_domain_names()
        if dataset == "DIMER":
            get_fn = get_dimer
            loader = DIMER_loader
            molecules = get_dimer_names()
            # take only dipeptides
            # molecules = [molecule for molecule in molecules if len(molecule) >= 5]
        if dataset == "AGG":
            get_fn = get_aggregate
            loader = AGG_loader
            molecules = get_AGG_structure_names()
        for name in tqdm(
            molecules, desc="Saving {} CG data...".format(dataset)
        ):
            pdb, filenames = get_fn(name, base_dir=opts["base_dir"])
            if dataset == "OPEP":
                mol_dictionary = get_mol_dictionary(
                    pdb,
                    pro_swap=opts["pro_swap"],
                    tag=opts["base_tag"] + "{:04d}".format(name),
                    embedding_strategy=data_generation_dict[
                        "embedding_strategy"
                    ],
                )
            else:
                mol_dictionary = get_mol_dictionary(
                    pdb,
                    pro_swap=opts["pro_swap"],
                    tag=opts["base_tag"] + "{}".format(name),
                    embedding_strategy=data_generation_dict[
                        "embedding_strategy"
                    ],
                )

            save_cg_coordforce(
                loader,
                mol_dictionary,
                filenames,
                save_dir=data_generation_dict["base_save_dir"],
                mapping=data_generation_dict["mapping"],
            )


def accumulate_data(data_generation_dict):
    """Accumulates data for the specified datasets
    and prior options

    Parameters
    ----------
    data_generation_dict:
        dictionary of dataset and prior options

    Returns
    -------
    collated_data:
        Collated data over the specified datasets and prior options
    """
    if data_generation_dict["seed"]:
        np.random.seed(data_generation_dict["seed"])
    data_list = []

    datasets = data_generation_dict["sub_datasets"]
    for dataset, opts in datasets.items():
        if opts["fit"] == False:
            continue
        if dataset == "OPEP":
            get_fn = get_opep
            molecules = np.arange(1100)
        if dataset == "CATH":
            get_fn = get_CATH
            molecules = get_CATH_domain_names()
        if dataset == "CATH_UNFOLDED":
            get_fn = get_CATH_unfolded
            molecules = get_CATH_domain_names()
        if dataset == "CATH_UNFOLDED_FINAL":
            get_fn = get_CATH_unfolded_final
            molecules = get_CATH_domain_names()
        if dataset == "DIMER":
            get_fn = get_dimer
            loader = DIMER_loader
            molecules = get_dimer_names()
            # take only dipeptides
            # molecules = [molecule for molecule in molecules if len(molecule) >= 5]
        if dataset == "AGG":
            get_fn = get_aggregate
            loader = AGG_loader
            molecules = get_AGG_structure_names()
        if opts["train_mols"] != None and opts["test_mols"] != None:
            train_mols = np.load(opts["train_mols"])
            test_mols = np.load(opts["test_mols"])
            if len(set(train_peptides).intersection(test_peptides)) != 0:
                raise RuntimeError(
                    "Overlap found between {} train/test sets".format(dataset)
                )
        else:
            train_mols, test_mols = train_test_split(
                molecules, test_size=0.2, shuffle=True
            )

        np.save(
            data_generation_dict["base_save_dir"]
            + "train_{}_{}.npy".format(
                dataset, data_generation_dict["prior_tag"]
            ),
            train_mols,
        )
        np.save(
            data_generation_dict["base_save_dir"]
            + "test_{}_{}.npy".format(
                dataset, data_generation_dict["prior_tag"]
            ),
            test_mols,
        )
        for name in tqdm(
            molecules, desc="Accumulating CG {} data".format(dataset)
        ):
            pdb, _ = get_fn(name, base_dir=opts["base_dir"])
            if dataset == "OPEP":
                mol_dictionary = get_mol_dictionary(
                    pdb,
                    pro_swap=opts["pro_swap"],
                    tag=opts["base_tag"] + "{:04d}".format(name),
                    embedding_strategy=data_generation_dict[
                        "embedding_strategy"
                    ],
                )
            else:
                mol_dictionary = get_mol_dictionary(
                    pdb,
                    pro_swap=opts["pro_swap"],
                    tag=opts["base_tag"] + "{}".format(name),
                    embedding_strategy=data_generation_dict[
                        "embedding_strategy"
                    ],
                )
            sub_data_list, prior_nls = process_accumulation(
                mol_dictionary,
                data_generation_dict,
                opts["acc_stride"],
            )
            with open(
                data_generation_dict["base_save_dir"]
                + mol_dictionary["tag"]
                + "_prior_nls_"
                + data_generation_dict["prior_tag"]
                + ".pkl",
                "wb",
            ) as pfile:
                pickle.dump(prior_nls, pfile)
            if name in train_mols:
                data_list += sub_data_list

    del sub_data_list, prior_nls
    # Collate the data ...
    print("Collating data...")
    datas, slices, _ = collate(
        data_list[0].__class__,
        data_list=data_list,
        increment=True,
        add_batch=True,
    )
    return datas


def fit_transferable_baseline_model(
    collated_data, data_generation_dict, temperature=300
):
    """Fits collated data according to the neighborlists
    and prior options that follow from the data generation dictionary

    Parameters
    ----------
    collated_data:
        Collated AtomicData instance
    data_generation_dict:
        dictionary of data generation options
    temperature:
        temperature of the data

    Returns
    -------
    prior_models:
        nn.ModuleDict of prior models
    all_stats:
        dictionary of statistics dictionaries for each prior fit
    """
    #:Boltzmann constant in kcal/mol/K
    kB = 0.0019872041
    beta = 1 / (temperature * kB)
    n_bond_bins = 1000
    n_angle_bins = 1000
    n_dihedral_bins = 1000
    n_omega_bins = 1000
    n_gamma_bins = 1000
    n_distances_bins = 5000
    distance_cutoff = data_generation_dict["distance_cutoff"]
    percentile = data_generation_dict["percentile"]

    bond_min = 1
    bond_max = 5
    angle_min = -1
    angle_max = 1
    dihedral_min = -np.pi
    dihedral_max = np.pi
    omega_min = -np.pi
    omega_max = np.pi
    gamma_1_min = -np.pi
    gamma_1_max = np.pi
    gamma_2_min = -np.pi
    gamma_2_max = np.pi

    distance_min = 0.0
    distance_max = 30.0

    dihedral_degs = 3
    pro_phi_degs = 1
    pro_omega_degs = 2
    non_pro_omega_degs = 1
    gamma_1_degs = 1
    gamma_2_degs = 1

    all_stats = {}
    prior_models = {}
    for tag in tqdm(
        collated_data.neighbor_list.keys(), desc="Fitting priors..."
    ):
        prior_type = tag.split("_")[-1]  # get last part of name
        if prior_type in ["bonds", "pbonds"]:
            stats = compute_statistics(
                collated_data,
                tag,
                beta=beta,
                TargetPrior=HarmonicBonds,
                nbins=n_bond_bins,
                bmin=bond_min,
                bmax=bond_max,
            )
            all_stats[tag] = stats
            prior_models[tag] = GradientsOut(
                GeneralBonds(stats, name=tag), targets="forces"
            )
        elif prior_type == "angles":
            stats = compute_statistics(
                collated_data,
                tag,
                beta=beta,
                TargetPrior=HarmonicAngles,
                nbins=n_angle_bins,
                bmin=angle_min,
                bmax=angle_max,
            )
            all_stats[tag] = stats
            prior_models[tag] = GradientsOut(
                GeneralAngles(stats, name=tag), targets="forces"
            )
        elif prior_type == "phi" or prior_type == "psi":
            if tag == "PRO_phi":  # PRO gets special treatment
                stats = compute_statistics(
                    collated_data,
                    tag,
                    beta=beta,
                    TargetPrior=Dihedral,
                    nbins=n_dihedral_bins,
                    bmin=dihedral_min,
                    bmax=dihedral_max,
                    target_fit_kwargs={
                        "n_degs": pro_phi_degs,
                        "constrain_deg": pro_phi_degs,
                    },
                )

                if data_generation_dict["embedding_strategy"] == "opep_termini":
                    print("correcting N-terminal {} stats...".format(tag))
                    # Make sure that all N-terminal gamma-1 parameters are the same as the bulk
                    for key in stats.keys():
                        if 25 in key:
                            n_n_atom_idxs = np.argwhere(
                                np.asarray(key) == 25
                            )  # identify N-term N atom involved
                            assert n_n_atom_idxs.shape == (1, 1)
                            bulk_key = list(key)
                            bulk_key[
                                n_n_atom_idxs[0][0]
                            ] = 21  # replace with bulk nitrogen
                            bulk_stats = stats[tuple(bulk_key)]
                            stats[
                                key
                            ] = bulk_stats  # replace N-terminal stats with bulk stats

                all_stats[tag] = stats
                prior = Dihedral(stats, n_degs=pro_phi_degs)
                for j in prior.named_buffers():
                    val = float(j[1][j[1] != 0][0])
                    if j[0].split("_")[0] == "k1":
                        k1s.append(val)
                    if j[0].split("_")[0] == "k2":
                        k2s.append(val)
                prior.name = tag
                prior_models[tag] = GradientsOut(prior, targets="forces")
            else:  # all other amino acids
                stats = compute_statistics(
                    collated_data,
                    tag,
                    beta=beta,
                    TargetPrior=Dihedral,
                    nbins=n_dihedral_bins,
                    bmin=dihedral_min,
                    bmax=dihedral_max,
                    target_fit_kwargs={
                        "n_degs": dihedral_degs,
                        "constrain_deg": dihedral_degs,
                    },
                )

                if data_generation_dict["embedding_strategy"] == "opep_termini":
                    print("correcting N-terminal {} stats...".format(tag))
                    # Make sure that all N-terminal gamma-1 parameters are the same as the bulk
                    for key in stats.keys():
                        if 25 in key:
                            n_n_atom_idxs = np.argwhere(
                                np.asarray(key) == 25
                            )  # identify N-term N atom involved
                            assert n_n_atom_idxs.shape == (1, 1)
                            bulk_key = list(key)
                            bulk_key[
                                n_n_atom_idxs[0][0]
                            ] = 21  # replace with bulk nitrogen
                            bulk_stats = stats[tuple(bulk_key)]
                            stats[
                                key
                            ] = bulk_stats  # replace N-terminal stats with bulk stats

                all_stats[tag] = stats
                prior = Dihedral(stats, n_degs=dihedral_degs)
                prior.name = tag
                prior_models[tag] = GradientsOut(prior, targets="forces")
        elif tag in ["pro_omega"]:  # use tag
            stats = compute_statistics(
                collated_data,
                tag,
                beta=beta,
                TargetPrior=Dihedral,
                nbins=n_omega_bins,
                bmin=omega_min,
                bmax=omega_max,
                target_fit_kwargs={
                    "n_degs": pro_omega_degs,
                    "constrain_deg": pro_omega_degs,
                },
            )
            # Here we have to patch some poor statistics based on GLY:
            # So we assign the fits from (22,21,23,22) to all
            # other keys that have a GLY CA (embedding 6)
            print("correcting GLY/PRO omega stats...")
            for key in stats.keys():
                if 6 in key:
                    stats[key] = stats[(22, 21, 23, 22)]
            all_stats[tag] = stats

            prior = Dihedral(stats, n_degs=pro_omega_degs)
            prior.name = tag
            prior_models[tag] = GradientsOut(prior, targets="forces")
        elif tag in ["non_pro_omega"]:  # use tag
            stats = compute_statistics(
                collated_data,
                tag,
                beta=beta,
                TargetPrior=Dihedral,
                nbins=n_omega_bins,
                bmin=omega_min,
                bmax=omega_max,
                target_fit_kwargs={
                    "n_degs": non_pro_omega_degs,
                    "constrain_deg": non_pro_omega_degs,
                },
            )
            # Here we have to patch some poor statistics based on GLY:
            # So we assign the fits from (22,21,23,22) to all
            # other keys that have a GLY CA (embedding 6)
            print("correcting GLY/Non-PRO omega stats...")
            for key in stats.keys():
                if 6 in key:
                    stats[key] = stats[(22, 21, 23, 22)]

            all_stats[tag] = stats
            prior = Dihedral(stats, n_degs=non_pro_omega_degs)
            prior.name = tag
            prior_models[tag] = GradientsOut(prior, targets="forces")

        elif tag in ["gamma_1"]:  # use tag for this instead
            stats = compute_statistics(
                collated_data,
                tag,
                beta=beta,
                TargetPrior=Dihedral,
                nbins=n_gamma_bins,
                bmin=gamma_1_min,
                bmax=gamma_1_max,
                target_fit_kwargs={
                    "n_degs": gamma_1_degs,
                    "constrain_deg": gamma_1_degs,
                },
            )

            if data_generation_dict["embedding_strategy"] == "opep_termini":
                print("correcting N-terminal {} stats...".format(tag))
                # Make sure that all N-terminal gamma-1 parameters are the same as the bulk
                for key in stats.keys():
                    if 25 in key:
                        n_n_atom_idxs = np.argwhere(
                            np.asarray(key) == 25
                        )  # identify N-term N atom involved
                        assert n_n_atom_idxs.shape == (1, 1)
                        bulk_key = list(key)
                        bulk_key[
                            n_n_atom_idxs[0][0]
                        ] = 21  # replace with bulk nitrogen
                        bulk_stats = stats[tuple(bulk_key)]
                        stats[
                            key
                        ] = bulk_stats  # replace N-terminal stats with bulk stats

            all_stats[tag] = stats
            prior = Dihedral(stats, n_degs=gamma_1_degs)
            prior.name = tag
            prior_models[tag] = GradientsOut(prior, targets="forces")

        elif tag in ["gamma_2"]:
            stats = compute_statistics(
                collated_data,
                tag,
                beta=beta,
                TargetPrior=Dihedral,
                nbins=n_gamma_bins,
                bmin=gamma_2_min,
                bmax=gamma_2_max,
                target_fit_kwargs={
                    "n_degs": gamma_2_degs,
                    "constrain_deg": gamma_2_degs,
                },
            )

            all_stats[tag] = stats
            prior = Dihedral(stats, n_degs=gamma_2_degs)
            prior.name = tag
            prior_models[tag] = GradientsOut(prior, targets="forces")

        elif tag == "non_bonded":
            if data_generation_dict["skip_non_bonded"] == True:
                continue
            stats = compute_statistics(
                collated_data,
                tag,
                beta=beta,
                TargetPrior=Repulsion,
                nbins=n_distances_bins,
                bmin=distance_min,
                bmax=distance_max,
                fit_from_values=True,
                target_fit_kwargs={
                    "percentile": percentile,
                    "cutoff": distance_cutoff,
                },
            )
            all_stats[tag] = stats
            repulsion = Repulsion(stats)
            repulsion.name = tag
            prior_models[tag] = GradientsOut(repulsion, targets="forces")
        else:
            raise RuntimeError("tag {} has no known prior type".format(tag))

    modules = torch.nn.ModuleDict(prior_models)
    full_prior_model = SumOut(modules, targets=["energy", "forces"])
    torch.save(
        full_prior_model,
        data_generation_dict["base_save_dir"]
        + "full_prior_model_"
        + data_generation_dict["prior_tag"]
        + ".pt",
    )
    return full_prior_model, all_stats


def make_delta_forces(
    mol_dictionary, full_prior_model, data_generation_dict, chunk=False
):
    """Helper function to produce and save delta forces

    Parameters
    ----------
    mol_dictionary:
        Dictionary of useful CG molecule information
    full_prior_model:
        Full mlcg prior model
    data_generation_dict:
        Data generation options dictionary
    """
    data_list = []
    types = mol_dictionary["types"]
    prior_nls = pickle.load(
        open(
            data_generation_dict["base_save_dir"]
            + mol_dictionary["tag"]
            + "_prior_nls_"
            + data_generation_dict["prior_tag"]
            + ".pkl",
            "rb",
        )
    )
    coords = np.load(
        data_generation_dict["base_save_dir"]
        + mol_dictionary["tag"]
        + "_cg_coords.npy"
    )
    forces = np.load(
        data_generation_dict["base_save_dir"]
        + mol_dictionary["tag"]
        + "_cg_forces.npy"
    )
    assert coords.shape == forces.shape
    num_atoms = coords.shape[1]
    num_frames = coords.shape[0]
    for i in range(num_frames):
        data = AtomicData.from_points(
            pos=torch.tensor(coords[i]),
            forces=torch.tensor(forces[i]),
            atom_types=torch.tensor(types),
            masses=None,
            neighborlist=prior_nls,
        )
        data_list.append(data)
    batch_size = 1000
    chunks = tuple(chunker(data_list, batch_size))
    for sub_data_list in tqdm(chunks, "Removing baseline forces"):
        _ = remove_baseline_forces(
            sub_data_list,
            full_prior_model.models,
        )
    delta_forces = []
    for i in range(num_frames):
        delta_forces.append(data_list[i].forces.detach().numpy())
    np.save(
        data_generation_dict["base_save_dir"]
        + mol_dictionary["tag"]
        + "_"
        + data_generation_dict["prior_tag"]
        + "_delta_forces.npy",
        np.concatenate(delta_forces, axis=0).reshape(*coords.shape),
    )


def save_delta_forces(data_generation_dict):
    """Produces and saves delta forces for specified data generation options

    Parameters
    ----------
    data_generation_dict:
        dictionary of data generation options
    """
    # load the specified prior file:
    full_prior_model = torch.load(
        data_generation_dict["base_save_dir"]
        + "full_prior_model_"
        + data_generation_dict["prior_tag"]
        + ".pt"
    )
    datasets = data_generation_dict["sub_datasets"]
    for dataset, opts in datasets.items():
        if dataset == "OPEP":
            get_fn = get_opep
            molecules = np.arange(1100)
        if dataset == "CATH":
            get_fn = get_CATH
            molecules = get_CATH_domain_names()
        if dataset == "CATH_UNFOLDED":
            get_fn = get_CATH_unfolded
            molecules = get_CATH_domain_names()
        if dataset == "CATH_UNFOLDED_FINAL":
            get_fn = get_CATH_unfolded_final
            molecules = get_CATH_domain_names()
        if dataset == "DIMER":
            get_fn = get_dimer
            loader = DIMER_loader
            molecules = get_dimer_names()
            # take only dipeptides
            # molecules = [molecule for molecule in molecules if len(molecule) >= 5]
        if dataset == "AGG":
            get_fn = get_aggregate
            loader = AGG_loader
            molecules = get_AGG_structure_names()

        for name in tqdm(
            molecules, desc="Producing CG {} delta forces...".format(dataset)
        ):
            pdb, _ = get_fn(name, base_dir=opts["base_dir"])
            if dataset == "OPEP":
                mol_dictionary = get_mol_dictionary(
                    pdb,
                    pro_swap=opts["pro_swap"],
                    tag=opts["base_tag"] + "{:04d}".format(name),
                    embedding_strategy=data_generation_dict[
                        "embedding_strategy"
                    ],
                )
            else:
                mol_dictionary = get_mol_dictionary(
                    pdb,
                    pro_swap=opts["pro_swap"],
                    tag=opts["base_tag"] + "{}".format(name),
                    embedding_strategy=data_generation_dict[
                        "embedding_strategy"
                    ],
                )
            make_delta_forces(
                mol_dictionary, full_prior_model, data_generation_dict
            )
        del mol_dictionary, pdb


def generate_targets(data_generation_dict):
    """Generates neighborlists for targets
    and prior options

    Parameters
    ----------
    data_generation_dict:
        dictionary of dataset and prior options
    """
    datasets = data_generation_dict["target_datasets"]
    for dataset, opts in datasets.items():
        if opts["save"] == True:
            if dataset == "CLN":
                get_fn = get_cln
                loader = CLN_loader
                molecules = get_cln_traj_names()

            if dataset == "BBA":
                get_fn = get_bba
                loader = BBA_loader
                molecules = get_bba_traj_names()

            if dataset == "Villin":
                get_fn = get_villin
                loader = villin_loader
                molecules = get_villin_traj_names()

            if dataset == "AMBER_CLN":
                get_fn = get_amber_cln
                loader = amber_cln_loader
                molecules = get_amber_cln_traj_names()

            if dataset == "TRPcage":
                get_fn = get_trpcage
                loader = TRPcage_loader
                molecules = get_trpcage_traj_names()

            if dataset == "WWdomain":
                get_fn = get_wwdomain
                loader = wwdomain_loader
                molecules = get_wwdomain_traj_names()

            if dataset == "DESRES_WWdomain":
                get_fn = get_desres_wwdomain
                loader = desres_wwdomain_loader
                molecules = get_desres_wwdomain_traj_names()

            if dataset == "Lambda":
                get_fn = get_lambda_repressor
                loader = lambda_repressor_loader
                molecules = get_lambda_repressor_traj_names()

            for name in tqdm(
                molecules, desc="Saving {} CG data...".format(dataset)
            ):
                pdb, filenames = get_fn(name, base_dir=opts["base_dir"])
                mol_dictionary = get_mol_dictionary(
                    pdb,
                    pro_swap=opts["pro_swap"],
                    tag=opts["base_tag"] + "{}".format(name),
                    embedding_strategy=data_generation_dict[
                        "embedding_strategy"
                    ],
                )

                save_cg_coordforce(
                    loader,
                    mol_dictionary,
                    filenames,
                    save_dir=data_generation_dict["base_save_dir"],
                    only_coords=(
                        True if dataset in ["DESRES_WWdomain"] else False
                    ),
                    mapping=data_generation_dict["mapping"],
                )

    datasets = data_generation_dict["target_datasets"]
    for dataset, opts in datasets.items():
        if dataset == "CLN":
            get_fn = get_cln
            molecules = get_cln_traj_names()
        if dataset == "BBA":
            get_fn = get_bba
            molecules = get_bba_traj_names()
        if dataset == "AMBER_CLN":
            get_fn = get_amber_cln
            molecules = get_amber_cln_traj_names()
        if dataset == "TRPcage":
            get_fn = get_trpcage
            molecules = get_trpcage_traj_names()
        if dataset == "Villin":
            get_fn = get_villin
            molecules = get_villin_traj_names()
        if dataset == "WWdomain":
            get_fn = get_wwdomain
            molecules = get_wwdomain_traj_names()
        if dataset == "DESRES_WWdomain":
            get_fn = get_desres_wwdomain
            molecules = get_desres_wwdomain_traj_names()
        if dataset == "Lambda":
            get_fn = get_lambda_repressor
            molecules = get_lambda_repressor_traj_names()

        for name in tqdm(
            molecules, desc="Accumulating CG {} data".format(dataset)
        ):
            pdb, _ = get_fn(name, base_dir=opts["base_dir"])
            mol_dictionary = get_mol_dictionary(
                pdb,
                pro_swap=opts["pro_swap"],
                tag=opts["base_tag"] + "{}".format(name),
                embedding_strategy=data_generation_dict["embedding_strategy"],
            )
        sub_data_list, prior_nls = process_accumulation(
            mol_dictionary,
            data_generation_dict,
        )
        with open(
            data_generation_dict["base_save_dir"]
            + dataset
            + "_prior_nls_"
            + data_generation_dict["prior_tag"]
            + ".pkl",
            "wb",
        ) as pfile:
            pickle.dump(prior_nls, pfile)
