import numpy as np
import warnings

__all__ = ["mol_split", "multimol_split"]
"""
Train-Validation splitting strategy
No cross validation concerns for the moment
we process everything in the sorted key order, so as to keep consistency over different runs/processes/machines
For each Metaset
  - if it only contains single molecule, then split the underlying data into train-val sets according to the ratio
  - if it contains multiple molecules, then split on the level of molecules, 
    (so that one molecule can only be in *either* train or val set, and the *total number of samples* is roughly according to the desired split)
"""
def _check_props(part_props):
    """Checking whether the provided ratio is valid."""
    props = np.array(list(part_props.values()))
    # all entries should be non-negative
    assert np.all(props >= 0.), "all proportions should be non-negative"
    # it shall not add up more than one (considering numerical error)
    assert np.sum(props) < 1 + 1e-4, "proportions add up more than 1."


# Method 1: splitting single-molecule metaset
# get n_frames from the single-molecule data, establish an index, randomly shuffle it (with a given random seed), and then 
# split the molecular data accordingly

def mol_split(n_frames, proportions={"train":0.8, "val":0.2},
              random_seed=42, verbose=False):
    _check_props(proportions)
    indices = np.arange(n_frames)
    rng = np.random.default_rng(random_seed)
    rng.shuffle(indices)
    split_indices = {}
    start_id = 0
    for part_name, part_prop in proportions.items():
        target_id = start_id + int(n_frames * part_prop)
        target_id = min(target_id, n_frames)
        split_indices[part_name] = indices[start_id:target_id]
        split_indices[part_name].sort()
        start_id = target_id
    if verbose:
        msg = "Actual split:\n"
        for part_name, part_indices in split_indices.items():
            msg += "- \"%s\": %d (%.1f%%)\n" % (part_name, len(part_indices), len(part_indices) / n_frames * 100.)
        print(msg)
    return split_indices

# Method 2: splitting multi-molecule metaset
# we consider all frames belong to each molecule as a whole in the split
# get n_frames from the single-molecule data, establish an index, 
# randomly shuffle it (with a given random seed), and then split the molecular data accordingly

def multimol_split(mol_count_dict, proportions={"train":0.8, "val":0.2},
                   random_seed=42, verbose=False):
    _check_props(proportions)
    mol_names = sorted(mol_count_dict)
    mol_frames = np.array([mol_count_dict[n] for n in mol_names])
    indices = np.arange(len(mol_names))
    rng = np.random.default_rng(random_seed)
    rng.shuffle(indices)
    # we go for a split whose train_ratio is closest to the desired value
    accum_frames = np.cumsum(mol_frames[indices])
    n_frames = accum_frames[-1]
    split_parts = {}
    def get_part_prop(part_name, count=False):
        if not part_name in split_parts:
            return 0.
        lb, ub = split_parts[part_name]
        return get_range_prop(lb, ub, count)
    def get_range_prop(lb, ub, count=False):
        frame_start = 0 if lb == 0 else accum_frames[lb - 1]
        frame_stop = 0 if ub == 0 else accum_frames[ub - 1]
        n_frames_part = frame_stop - frame_start
        if count:
            return n_frames_part
        else:
            return n_frames_part / n_frames
    split_prop = 0.
    last_split_index = 0
    for (part, prop) in sorted(list(proportions.items()), key=lambda tup: -tup[1]):
        target_prop = split_prop + prop
        split_index = np.argmin(np.abs(accum_frames - n_frames * target_prop)) + 1
        split_parts[part] = (last_split_index, split_index)
        split_prop += get_range_prop(last_split_index, split_index)
        last_split_index = split_index

    if any((get_part_prop(p) == 0. for p in proportions)):
        warnings.warn("No solution with a partition ratio close to demanded is found:"
                     " at least one partition is empty.")
    # transcribe the results
    split_mols = {}
    for part in split_parts:
        lb, ub = split_parts[part]
        split_mols[part] = [mol_names[i] for i in sorted(indices[lb:ub])]
    if verbose:
        msg = "Actual split:\n"
        n_frames = accum_frames[-1]
        for part_name, part_range in split_parts.items():
            n_frames_part = get_part_prop(part_name, count=True)
            msg += f"- `{part_name}`: {n_frames_part} ({n_frames_part / n_frames * 100.:.1f}%)\n"
        print(msg)
    return split_mols

