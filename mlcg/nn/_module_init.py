import torch


def init_xavier_uniform(module: torch.nn.Module) -> None:
    """initialize (in place) weights of the input module using xavier uniform.
    Works only on `torch.nn.Linear` at the moment and the bias are set to 0.

    Parameters
    ----------
    module : torch.nn.Module
        a torch module
    """
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0.)
