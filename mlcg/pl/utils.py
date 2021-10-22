import torch
import pytorch_lightning.plugins as plp
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from typing import List, Optional, Union, Any, Dict

class SingleDevicePlugin(plp.SingleDevicePlugin):
    """Plugin that handles communication on a single device."""

    def __init__(self, device: str = "cpu"):
        super(SingleDevicePlugin, self).__init__(torch.device(device))

class DDPPlugin(plp.DDPPlugin):
    """Plugin that handles communication on a single device."""
    def __init__(self, parallel_devices: Optional[List[int]] = None, num_nodes: Optional[int] = None, cluster_environment: ClusterEnvironment = None, sync_batchnorm: Optional[bool] = None, ddp_comm_state: Optional[object] = None, ddp_comm_hook: Optional[callable] = None, ddp_comm_wrapper: Optional[callable] = None, **kwargs: Union[Any, Dict[str, Any]]) -> None:
        if parallel_devices is not None:
            parallel_devices = [torch.device(f'cuda:{ii}') for ii in  parallel_devices]
        super().__init__(parallel_devices=parallel_devices, num_nodes=num_nodes, cluster_environment=cluster_environment, sync_batchnorm=sync_batchnorm, ddp_comm_state=ddp_comm_state, ddp_comm_hook=ddp_comm_hook, ddp_comm_wrapper=ddp_comm_wrapper, **kwargs)

    # def __init__(self, device_ids: List[int] = "cpu"):
    #     super(SingleDevicePlugin, self).__init__(torch.device(device))