import yaml
import argparse
import numpy as np
import torch
import inspect
from sklearn.model_selection import train_test_split

try:
    from pytorch_lightning.trainer.states import RunningStage
except ImportError:
    # compatibility for PyTorch lightning versions < 1.2.0
    RunningStage = None

def is_notebook():
    from IPython import get_ipython
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def load_yaml(fn):
    """ load a yaml file
    """
    with open(fn, 'r') as f:
        data = yaml.safe_load(f)
    return data


def train_val_test_split(dset_len,val_ratio,test_ratio, seed=None, order=None):
    shuffle = True if order is None else False
    valtest_ratio = val_ratio + test_ratio
    idx_train = list(range(dset_len))
    idx_test = []
    idx_val = []
    if valtest_ratio > 0 and dset_len > 0:
        idx_train, idx_tmp = train_test_split(range(dset_len), test_size=valtest_ratio, random_state=seed, shuffle=shuffle)
        if test_ratio == 0:
            idx_val = idx_tmp
        elif val_ratio == 0:
            idx_test = idx_tmp
        else:
            test_val_ratio = test_ratio / (test_ratio + val_ratio)
            idx_val, idx_test = train_test_split(idx_tmp, test_size=test_val_ratio, random_state=seed, shuffle=shuffle)

    if order is not None:
        idx_train = [order[i] for i in idx_train]
        idx_val = [order[i] for i in idx_val]
        idx_test = [order[i] for i in idx_test]

    return np.array(idx_train), np.array(idx_val), np.array(idx_test)


def make_splits(dataset_len, val_ratio, test_ratio, seed=None, filename=None, splits=None, order=None):
    if splits is not None:
        splits = np.load(splits)
        idx_train = splits['idx_train']
        idx_val = splits['idx_val']
        idx_test = splits['idx_test']
    else:
        idx_train, idx_val, idx_test = train_val_test_split(
            dataset_len, val_ratio, test_ratio, seed, order
        )

    if filename is not None:
        np.savez(filename, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

    return torch.from_numpy(idx_train), torch.from_numpy(idx_val), torch.from_numpy(idx_test)


