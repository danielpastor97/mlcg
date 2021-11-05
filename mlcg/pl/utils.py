from numpy import isin
import torch
import pytorch_lightning.plugins as plp
from pytorch_lightning.plugins.environments.cluster_environment import (
    ClusterEnvironment,
)
from typing import List, Optional, Union, Any, Dict

from mlcg.nn import prior

from .model import PLModel
from ..nn import SumOut


def extract_model_from_checkpoint(checkpoint_path, hparams_file):
    plmodel = PLModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path, hparams_file=hparams_file
    )
    return plmodel.get_model()


def merge_priors_and_checkpoint(
    checkpoint_path: str,
    priors: Union[str, torch.nn.ModuleDict],
    hparams_file: Optional[str] = None,
) -> torch.nn.Module:
    """load prior models and trained model from a checkpoint and merge them
    into a :ref:`mlcg.nn.SumOut` module.

    Parameters
    ----------
    checkpoint_path :
        full path to the checkpoint file
    priors :
        If :obj:`torch.nn.ModuleDict`, it should be the collection of priors used as a baseline for training the ML model.
        If :obj:`str`, it should be a full path to the file hoding the priors.
    hparams_file :
        full path to the hyper parameter file associated with the checkpoint file. It is typically not necessary to provide it.

    Returns
    -------
        a :ref:`mlcg.nn.SumOut` module containing the trained model with the priors
    """

    ml_model = extract_model_from_checkpoint(checkpoint_path, hparams_file)
    if isinstance(priors, str):
        prior_model = torch.load(priors)
    else:
        prior_model = priors
    # prior_model should be a ModuleDict
    prior_model[ml_model.name] = ml_model
    model = SumOut(models=prior_model)
    return model


class SingleDevicePlugin(plp.SingleDevicePlugin):
    """Plugin that handles communication on a single device."""

    def __init__(self, device: str = "cpu"):
        super(SingleDevicePlugin, self).__init__(torch.device(device))


class DDPPlugin(plp.DDPPlugin):
    """Plugin that handles communication on a single device."""

    def __init__(
        self,
        parallel_devices: Optional[List[int]] = None,
        num_nodes: Optional[int] = None,
        cluster_environment: ClusterEnvironment = None,
        sync_batchnorm: Optional[bool] = None,
        ddp_comm_state: Optional[object] = None,
        ddp_comm_hook: Optional[callable] = None,
        ddp_comm_wrapper: Optional[callable] = None,
        **kwargs: Union[Any, Dict[str, Any]],
    ) -> None:
        if parallel_devices is not None:
            parallel_devices = [
                torch.device(f"cuda:{ii}") for ii in parallel_devices
            ]
        super().__init__(
            parallel_devices=parallel_devices,
            num_nodes=num_nodes,
            cluster_environment=cluster_environment,
            sync_batchnorm=sync_batchnorm,
            ddp_comm_state=ddp_comm_state,
            ddp_comm_hook=ddp_comm_hook,
            ddp_comm_wrapper=ddp_comm_wrapper,
            **kwargs,
        )

    # def __init__(self, device_ids: List[int] = "cpu"):
    #     super(SingleDevicePlugin, self).__init__(torch.device(device))
