import torch
from pytorch_lightning.plugins import (
    SingleDevicePlugin as SingleDevicePlugin_pl,
)


class SingleDevicePlugin(SingleDevicePlugin_pl):
    """Plugin that handles communication on a single device."""

    def __init__(self, device: str = "cpu"):
        super(SingleDevicePlugin, self).__init__(torch.device(device))
