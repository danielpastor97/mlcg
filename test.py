from matplotlib import pyplot as plt
import torch
import mdtraj
import sys
import numpy as np
from os.path import join
from tqdm.notebook import tqdm
from copy import deepcopy

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.data.collate import collate, _collate

from mlcg.datasets.chignolin import ChignolinDatasetWithNewPriors

sys.path.insert(0, "./")
from mlcg.data.atomic_data import AtomicData
from mlcg.geometry.topology import Topology
from mlcg.cg.projection import build_cg_matrix, build_cg_topology, CA_MAP
from mlcg.neighbor_list.neighbor_list import (
    atomic_data2neighbor_list,
    validate_neighborlist,
)
from mlcg.datasets import ChignolinDataset
from mlcg.geometry.statistics import compute_statistics

if __name__ == "__main__":
    dataset = ChignolinDataset("/net/storage/clarkt/chignolin")

    print(dataset.data.baseline_forces)

    dataset0 = ChignolinDatasetWithNewPriors("/net/storage/clarkt/chignolin")
    dataset0.process()
