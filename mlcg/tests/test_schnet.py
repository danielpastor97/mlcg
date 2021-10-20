import numpy as np
import torch
from torch import nn
from mlcg.nn.schnet import *
from mlcg.data.atomic_data import AtomicData


#### Test Data ####

test_data = ...

###################


"""
Test idea list...

1. Check to see if the cutoff warnings are generated in schnet.__init__()
2. Test to make sure that self_inclusion edges are generated
3. Test CFConv output (torch vs. numpy)
4. Test max_num_neighbors
5. Test helper function
"""
