# from matplotlib import pyplot as plt
# import torch
# import mdtraj
# import sys
# import numpy as np
# from os.path import join
# from tqdm.notebook import tqdm
# from copy import deepcopy

# from torch_geometric.loader import DataLoader
# from torch_geometric.data import Data
# from torch_geometric.data.collate import collate, _collate

# sys.path.insert(0, "./")
# from mlcg.data.atomic_data import AtomicData
# from mlcg.geometry.topology import Topology
# from mlcg.cg.projection import build_cg_matrix, build_cg_topology, CA_MAP
# from mlcg.neighbor_list.neighbor_list import (
#     atomic_data2neighbor_list,
#     validate_neighborlist,
# )
# from mlcg.datasets import ChignolinDataset
# from mlcg.geometry.statistics import compute_statistics

# if __name__ == "__main__":
#     dataset = ChignolinDataset("/local_scratch/musil/datasets/test")

#     print(dataset.data.baseline_forces)

from typing import Optional, Tuple
import yaml


def instantiate_class(class_path):
    kwargs = init.get("init_args", {})
    if not isinstance(args, tuple):
        args = (args,)
    class_module, class_name = init["class_path"].rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    args_class = getattr(module, class_name)
    return args_class(*args, **kwargs)


tag = "!python/typing"
aa = Tuple
dd = lambda dumper, data: dumper.represent_str(tag + "%s" % str(data))
cc = lambda loader, node: eval(node)
yaml.add_representer(type(Tuple), dd)
tag = "!%s" % str(aa)
print(tag)
yaml.add_constructor(tag, cc)

print(yaml.dump(aa))
# yaml.load("name: !typing Tuple", yaml.Loader)
