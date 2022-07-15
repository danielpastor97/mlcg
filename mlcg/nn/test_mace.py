import pytest

try:
    import mace
    from mlcg.nn.mace_interface import MACEInterface
except:
    pytest.skip("MACE installation not found...", allow_module_level=True)

import torch
import numpy as np
from torch_geometric.data.collate import collate

from mlcg.nn.schnet import StandardSchNet
from mlcg.nn.gradients import GradientsOut
from mlcg.geometry import Topology
from mlcg.data.atomic_data import AtomicData
from mlcg.data._keys import ENERGY_KEY, FORCE_KEY

from mlcg.nn.test_schnet import MolDatabase

database = MolDatabase()

mace_config = {
    "r_max": 10,
    "num_bessel": 10,
    "num_polynomial_cutoff": 5,
    "max_ell": 2,
    "interaction_cls": mace.modules.blocks.RealAgnosticResidualInteractionBlock,
    "interaction_cls_first": mace.modules.blocks.RealAgnosticInteractionBlock,
    "num_interactions": 1,
    "num_elements": len(database.atomic_numbers),
    "hidden_irreps": "256x0e",
    "MLP_irreps": "16x0e",
    "avg_num_neighbors": 9,
    "correlation": 4,
    "gate": torch.nn.Tanh,
    "atomic_numbers": database.atomic_numbers,
}


@pytest.mark.parametrize(
    "collated_data, out_keys, expected_shapes",
    [
        (
            database.collated_data,
            [ENERGY_KEY, FORCE_KEY],
            [database.energy_shape, database.force_shape],
        )
    ],
)
def test_prediction(collated_data, out_keys, expected_shapes):
    """Test to make sure that the output dictionary is properly populated
    and that the correspdonding shapes of the outputs are correct given the
    requested gradient targets.
    """
    print(collated_data.pos.dtype)

    test_mace = MACEInterface(config=mace_config, gate=torch.nn.Tanh())
    model = GradientsOut(test_mace, targets=FORCE_KEY).float()
    collated_data = model(collated_data)
    assert len(collated_data.out) != 0
    assert "mace" in collated_data.out.keys()
    for key, shape in zip(out_keys, expected_shapes):
        assert key in collated_data.out[model.name].keys()
        assert collated_data.out[model.name][key].shape == shape
