"""Python classes for processing data stored in HDF5 format.

HDF5 format benefits the dataset management for mlcg-tools when training/validation involves multiple molecules of vastly different lengths and when parallelization is used.
The main features are:
1. The internal structure mimics the hierarchy of the dataset itself, such that we don't have to replicate it on filesystem.
2. we don't have to actively open all files in the process
3. we can transparently load only the necessary part of the dataset to the memory

This file contains the python data structures for handling the HDF5 file and its content, i.e., the coordinates, forces and embedding vectors for multiple CG molecules.
An example HDF5 file structure and correponding class types after loading:
/ (HDF-group, => `H5Dataset._h5_root`)
├── OPEP (HDF-group =(part, according to "partition_options")=> `Metaset` in a `Partition`)
│   ├── opep_0000 (HDF-group, => `MolData`)
│   │   ├── attrs (HDF-attributes of the molecule "/OPEP/opep_0000")
│   │   │   ├── cg_embeds (a 1-d numpy.int array)
│   │   │   ├── N_frames (int, number of frames = size of cg_coords on axis 0)
│   │   │   ... (optional, e.g., cg_top, cg_pdb, etc that corrsponds to the molecule)
│   │   ├── cg_coords (HDF-dataset of the molecule "/OPEP/opep_0000", 3-d numpy.float32 array)
│   │   └── cg(_delta)_forces (HDF-dataset of the molecule "/OPEP/opep_0000", 3-d numpy.float32 array)
│   ... (other molecules in "/OPEP")
├── CATH (HDF-group ="partition_options"=> `Metaset` in a `Partition`)
...

> Data structure `MolData` is the basic brick of the dataset.
It holds embeds, coords and forces of certain number of frames for a single molecule.
.sub_sample: method for performing indexing on the frames.

> Data strcuture `Metaset` holds multiple molecules and joins their frame indices together.
One can access frames of underlying MolData objects with a continuous index.
The idea is that the molecules in a `Metaset` have similar numbers of CG beads, such that the sample requires similar processing time when passing through a neural network.
When this rule is enforced, it will allow automatic balancing of batch composition during training and validation.
- .create_from_hdf5_group: initiate a Metaset by loading data from given HDF-group.
mol_list, detailed_indices, hdf_key_mapping, stride and parallel can control which subset is loaded to the Metaset (details see the description of "H5Dataset")
- .trim_down_to: drop frames of underlying for obtaining a subset with desired number of frames.
The indices of frames to be dropped are controlled by a random number generator. When the parameter "random_seed" and the MolData order and number of frames are the same, the frames to be dropped will be reproducible.
- len() (__len__): return total number of samples
- [] (__get_item__): infer the molecule and get the corresponding frame according to the given unified index. Return value is grouped by an `AtomicData` object.


> Data structure `Partition` can hold multiple Metaset for training and validation purposes.
Its main function is to automatic adjust (subsample) the Metaset(s) and the underlying MolData to form a balanced data source, from which a certain number of batches can be drawn. Each batch will contain a pre-given number of samples from each Metaset.
One or several `Partition`s can be created to load part of the same HDF5 file into the memory.
The exact content inside a `Partition` object is controlled by the `partition_options` as a parameter for initializing a `H5Dataset`.

> Data strcuture `H5Dataset` opens a HDF5 file and establish the partitions.
`partition_options` describe what partitions to create and what content to load into them. (Detailed description and examples are as followed.)
`loading_options` mainly deal with the HDF key mapping (which datasets/attributes corresponds to the coordinates, forces and embeddings) as well as (optionally) the information for correctly split the dataset in a parallel training/validation.

An example "partition_options" (as a Python Mappable (e.g., dict)):
{
    "train": {
        "metasets": {
            "OPEP": {
                "molecules": [
                    "opep_0000",
                    "opep_0001",
                    ...
                ],
                "stride": 1, # optional, default 1
                "detailed_indices": { # optional, providing the indices of frames to work with (before striding and splitting for parallel processes).
                    # If detailed_indices are not provided for a given molecule, then it is equivalent to np.arange(N_frames)
                    "opep_0000": [1, 3, 5, 7, 9, ...],
                },
            },
            "CATH": {
                "molecules": [
                    "cath_1b43A02",
                    ...
                ],
                "stride": 1, # optional
                "detailed_indices": {}, # optional
            }
        },
        "batch_size": {
            # each batch will contain 24 samples from
            # The two metasets will be trimmed down according to this ratio
            # so optimally it should be proportional to the ratio of numbers of frames in the metasets.
            "OPEP": 24,
            "CATH": 6,
        },
        "subsample_random_seed": 42, # optional, default 42. Random seed for trimming down the frames.
        "max_epoch_samples": None, # optional, default None. For setting a desired dataset size.
        # Works by subsampling the dataset after it is loaded with the given stride.
    },
    "val": { # similar
    }
}

An example "loading_options" (as a Python Mappable (e.g., dict)):
{
    "hdf_key_mapping": {
        # keys for reading CG data from HDF5 groups
        "embeds": "attrs:cg_embeds",
        "coords": "cg_coords",
        "forces": "cg_delta_forces",
    },
    "parallel": { # optional, default rank=0 and world_size=1 (single process).
        # For split the dataset evenly and load only the necessary parts to each process in a parallelized train/val setup
        "rank": 0, # which process is this
        "world_size": 1, # how many processes are there
    }
}


> Class `H5PartitionDataLoader` mimics the pytorch data loader and can generate training/validation batches with fixed proportion of samples from several Metasets in a Partition.
The proportion and batch size is defined when the partition is initialized.
When the molecules in each Metaset have similar embedding vector lengths, the processing of output batches will require a more or less fixed size of VRAM on GPU, which can benefit the

Note:
1. Usually in a train-val split, each molecule goes to either the train or the test partition.
In some special cases (e.g., non-transferable training) one molecule can be part of both partitions.
In this situation, "detailed_indices" can be set to assign the corresponding frames to the desired partitions.

"""
import h5py
import numpy as np
import torch
import typing
import itertools
from torch_geometric.loader.dataloader import Collater as PyGCollater

from mlcg.data import AtomicData


class MolData:
    """Data-holder for coordinates, forces and embedding vector of a molecule, e.g., opep_0000."""

    def __init__(
        self,
        name: str,
        embeds: np.ndarray,
        coords: np.ndarray,
        forces: np.ndarray,
    ):
        # TODO: check input consistency (e.g., number of CG beads and frames)
        self._name = name
        self._embeds = embeds
        self._coords = coords
        self._forces = forces

    @property
    def name(self):
        return self._name

    @property
    def embeds(self):
        return self._embeds

    @property
    def coords(self):
        return self._coords

    @property
    def forces(self):
        return self._forces

    @property
    def n_frames(self):
        return self.coords.shape[0]

    @property
    def n_beads(self):
        return self.coords.shape[1]

    def __repr__(self):
        return (f"""MolData(name={self.name}", N_beads={self.n_beads}, N_frames={self.n_frames})"""
        )

    def sub_sample(self, indices):
        self._coords = self._coords[indices]
        self._forces = self._forces[indices]


class MetaSet:
    def __init__(self, name):
        self.name = name
        self._mol_dataset = []
        self._mol_map = {}
        self._n_mol_samples = []
        self._cumulate_indices = [0]

    @staticmethod
    def create_from_hdf5_group(
        hdf5_group,
        mol_list,
        detailed_indices=None,
        stride=1,
        hdf_key_mapping={
            "embeds": "attrs:cg_embeds",
            "coords": "cg_coords",
            "forces": "cg_delta_forces",
        },
        parallel={
            "rank": 0,
            "world_size": 1,
        },
    ):
        def select_for_rank(length_or_indices):
            """Return a slicing for loading the necessary data from the HDF dataset."""
            from numbers import Integral

            if isinstance(length_or_indices, Integral):  # int or np.int*
                indices = None
                length = length_or_indices
            else:
                indices = length_or_indices
                length = len(indices)
            rank = parallel.get("rank", 0)
            world_size = parallel.get("world_size", 1)
            if rank < 0 or rank >= world_size:
                raise ValueError(
                    "Rank %d is invalid given the world_size %d."
                    % (rank, world_size)
                )
            if length < world_size:
                raise ValueError(
                    "Length %d is too short given the world_size %d."
                    % (length, world_size)
                )
            residue = length % world_size
            if indices is not None:
                return indices[(residue + rank) :: (world_size * stride)]
            else:
                return np.s_[(residue + rank) :: (world_size * stride)]

        def retrieve_hdf(hdf_grp, hdf_key):
            """Unified hdf retriever for attributes and dataset."""

            def is_attr(hdf_key):
                return hdf_key[:6] == "attrs:"

            if is_attr(hdf_key):
                return hdf_grp.attrs[hdf_key[6:]]
            else:
                return hdf_grp[hdf_key]

        output = MetaSet(hdf5_group.name.split("/")[-1])
        keys = hdf_key_mapping
        for mol_name in mol_list:
            if mol_name not in hdf5_group:
                raise KeyError(
                    "`mol_list` includes moleucle %s,"
                    " which is not in the h5 group at path %s"
                    % (mol_name, hdf5_group.name)
                )
            embeds = retrieve_hdf(hdf5_group[mol_name], keys["embeds"])
            if (
                detailed_indices is not None
                and detailed_indices.get(mol_name) is not None
            ):
                # this molecule need to be splitted according to the detailed indices
                par_range = detailed_indices[mol_name]
                split_per_index = True
            else:
                par_range = retrieve_hdf(hdf5_group[mol_name], "attrs:N_frames")
                split_per_index = False
            selection = select_for_rank(par_range)
            if not split_per_index:
                coords = retrieve_hdf(hdf5_group[mol_name], keys["coords"])[
                    selection
                ]
                forces = retrieve_hdf(hdf5_group[mol_name], keys["forces"])[
                    selection
                ]
            else:
                # For large dataset it is usually quicker to first load everything
                # and then perform indexing in memory
                coords = retrieve_hdf(hdf5_group[mol_name], keys["coords"])[:][
                    selection
                ]
                forces = retrieve_hdf(hdf5_group[mol_name], keys["forces"])[:][
                    selection
                ]
            output.insert_mol(MolData(mol_name, embeds, coords, forces))
        return output

    def insert_mol(self, mol_data):
        self._mol_dataset.append(mol_data)
        self._mol_map[mol_data.name] = self.n_mol - 1
        self._update_info()

    def trim_down_to(self, target_n_samples, random_seed=42):
        """Trimming all datasets randomly to reach the target number of samples"""
        if target_n_samples > self.n_total_samples or target_n_samples <= 0:
            raise ValueError("Target number of samples invalid")
        rng = np.random.default_rng(random_seed)
        # get the indices of samples to keep
        indices = rng.choice(
            self.n_total_samples, size=target_n_samples, replace=False
        )
        indices.sort()
        # get the indices for each molecule
        base_index = 0
        left_bound = 0
        for i, next_start_index in enumerate(self._cumulate_indices):
            right_bound = np.searchsorted(indices, next_start_index)
            # subsample the molecular record
            self._mol_dataset[i].sub_sample(
                indices[left_bound:right_bound] - base_index
            )
            base_index = next_start_index
            left_bound = right_bound
        # update the n_mol_samples and cumulate_indices
        self._update_info()

    def _update_info(self):
        self._n_mol_samples = np.array(
            [mol_d.n_frames for mol_d in self._mol_dataset]
        )
        self._cumulate_indices = np.cumsum(self._n_mol_samples)

    @property
    def n_mol(self):
        return len(self._mol_dataset)

    @property
    def n_total_samples(self):
        return self._cumulate_indices[-1]

    @property
    def n_mol_samples(self):
        return self._n_mol_samples

    def get_mol_data_by_name(self, mol_name):
        index = self._mol_map.get(mol_name, None)
        if index is not None:
            return self._mol_dataset[index]

    def __getitem__(self, idx):
        dataset_id, data_id = self._locate_idx(idx)
        return AtomicData.from_points(
            pos=self._mol_dataset[dataset_id].coords[data_id],
            forces=self._mol_dataset[dataset_id].forces[data_id],
            atom_types=self._mol_dataset[dataset_id].embeds,
        )

    def _locate_idx(self, idx):
        full_length = self._cumulate_indices[-1]
        if idx < 0:
            idx += full_length
        if idx < 0 or idx >= full_length:
            raise IndexError("Given idx is outside of the dataset.")
        # find in which dataset it resides
        dataset_id = np.searchsorted(self._cumulate_indices, idx, side="right")
        # locate the position within the dataset
        if dataset_id == 0:
            datapt_id = idx
        else:
            datapt_id = idx - self._cumulate_indices[dataset_id - 1]
        return dataset_id, datapt_id

    def __len__(self):
        return self.n_total_samples

    def __repr__(self):
        return f"Metaset with {self.n_mol} molecules and {self.n_total_samples} samples"


class Partition:
    """Contain several metasets for a certain purpose, e.g., training."""

    def __init__(self, name):
        self.name = name
        self._metasets = {}
        self._sample_ready = False

    def add_metaset(self, metaset_name, metaset):
        self._metasets[metaset_name] = metaset
        self._sample_ready = False

    def get_metaset(self, metaset_name):
        assert self._sample_ready, "Sampling setup needs to be done first."
        return self._metasets.get(metaset_name, None)

    def get_metasets(self):
        assert self._sample_ready, "Sampling setup needs to be done first."
        return self._metasets

    def sampling_setup(
        self,
        batch_sizes: typing.Dict[str, MetaSet],
        max_epoch_samples=None,
        random_seed=42,
    ):
        """Calculate the number of batches available for an epoch according to
        the batch size for each metaset and optionally the maximum number of
        samples in an epoch. The molecular dataset will be trimmed accordingly.
        """
        if set(batch_sizes) != set(self._metasets):
            raise ValueError(
                "Input `batch_sizes` shall have exactly the same keys"
                " as the metasets in this H5Dataset: "
                + str(list(self._metasets.keys()))
            )
        self._batch_sizes = batch_sizes
        # calculating the number of batches
        sample_sizes = self.sample_sizes
        max_batches = [
            int(sample_sizes[k] // batch_sizes[k]) for k in batch_sizes
        ]
        if max_epoch_samples is not None:
            max_batches.append(max_epoch_samples // sum(batch_sizes.values()))
        n_batches = min(max_batches)
        # trimming down the number of samples in each metaset accordingly
        for k in self._metasets:
            self._metasets[k].trim_down_to(
                n_batches * batch_sizes[k], random_seed=random_seed
            )
        self._sample_ready = True

    @property
    def batch_sizes(self):
        assert self._sample_ready, "Sampling setup needs to be done first."
        return self._batch_sizes

    @property
    def sample_sizes(self):
        return {k: self._metasets[k].n_total_samples for k in self._metasets}

    def __repr__(self):
        return "Partition `%s` with Metasets:\n" % self.name + "\n".join(
            [
                '- "%s": %d samples' % (k, v)
                for k, v in self.sample_sizes.items()
            ]
        )


class H5Dataset:
    """The top-level class for handling multiple datasets contained in a HDF5 file."""

    def __init__(self, h5_path, partition_options, loading_options):
        self._h5_path = h5_path
        self._h5_root = h5py.File(h5_path, "r")
        self._metaset_entries = {}
        # ^ dict containing all metasets in the HDF5 file
        self._partitions = {}  # dict containing the configured metasets
        self._sample_ready = False

        # processing the hdf5 file
        for metaset_name in self._h5_root:
            self._metaset_entries[metaset_name] = self._h5_root[metaset_name]

        self._create_partitions(partition_options, loading_options)

    def _create_partitions(self, partition_options, loading_options):
        ## TODO: sanity check of the given dictionary
        hdf_key_mapping = loading_options.get("hdf_key_mapping")
        parallel = loading_options.get(
            "parallel",
            {
                "rank": 0,
                "world_size": 1,
            },
        )  # if no parallel entry, then target for single process
        for part_name in partition_options:
            ## create a partition
            part = Partition(part_name)
            part_info = partition_options[part_name]
            # TODO: check option consistency
            for metaset_name in part_info["metasets"]:
                if metaset_name not in self._metaset_entries:
                    raise KeyError(
                        f"`partition_options` refer to metaset {part},"
                        f" which is not in the h5 dataset at path {self._h5_path}"
                    )
                mol_list = part_info["metasets"][metaset_name]["molecules"]
                detailed_indices = part_info["metasets"][metaset_name].get(
                    "detailed_indices", None
                )
                stride = part_info["metasets"][metaset_name].get("stride", 1)
                part.add_metaset(
                    metaset_name,
                    MetaSet.create_from_hdf5_group(
                        self._metaset_entries[metaset_name],
                        mol_list,
                        detailed_indices,
                        stride=stride,
                        hdf_key_mapping=hdf_key_mapping,
                        parallel=parallel,
                    ),
                )
            ## trim the metasets to fit the need of sampling
            random_seed = part_info.get("subsample_random_seed", 42)
            max_epoch_samples = part_info.get("max_epoch_samples", None)
            if max_epoch_samples is not None and parallel["world_size"] > 1:
                max_epoch_samples = int(
                    max_epoch_samples // parallel["world_size"]
                )
            part.sampling_setup(
                part_info["batch_sizes"],
                max_epoch_samples,
                random_seed=random_seed,
            )
            self._partitions[part_name] = part

    def __del__(self):
        self._h5_root.close()

    @property
    def metaset_hdf_entries(self):
        return self._metaset_entries

    def partition(self, partition_name):
        return self._partitions.get(partition_name, None)

    def __repr__(self):
        return (
            'H5Dataset:\nPath: "%s"\nPartitions:\n' % self._h5_path
            + "\n".join(['- "' + str(part) + '"' for part in self._partitions])
        )


class H5PartitionDataLoader:
    """Load batches from one or multiple Metasets in a Partition."""

    def __init__(
        self,
        data_partition,
        collater_fn=PyGCollater(None, None),
        pin_memory=False,
    ):
        self._data_part = data_partition
        self._metasets = []
        self._samplers = []
        self._pin_memory = pin_memory
        for (metaset_name, batch_size) in data_partition.batch_sizes.items():
            metaset = data_partition.get_metaset(metaset_name)
            # ^ automatically checks whether the partition is sample_ready()
            self._metasets.append(metaset)
            s = torch.utils.data.RandomSampler(metaset)
            batch_s = torch.utils.data.BatchSampler(s, batch_size, True)
            self._samplers.append(batch_s)
        self._collater_fn = collater_fn

    def __iter__(self):
        self._sampler_iters = []
        for s in self._samplers:
            self._sampler_iters.append(iter(s))
        return self

    def __next__(self):
        # throw "StopIteration" when any of the sampler finishes iteration
        try:
            merged_samples = tuple(
                itertools.chain.from_iterable(
                    (
                        (dts[i] for i in next(si))
                        for dts, si in zip(self._metasets, self._sampler_iters)
                    )
                )
            )
        except RuntimeError as e:  # catch RuntimeError raised by drained sampler iterater
            raise StopIteration()
        output = self._collater_fn(merged_samples)
        if self._pin_memory:
            output = output.pin_memory()
        return output

    def __len__(self):
        return min([len(s) for s in self._samplers])
