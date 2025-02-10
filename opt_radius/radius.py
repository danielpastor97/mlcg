from typing import Optional, Tuple
import torch
from os import path

# This means we are compiling every time when loading this module
# if there is no cache. Ideal for debugging but not release
# TODO: set a proper "TORCH_CUDA_ARCH_LIST" to remove the warning
# ref: https://github.com/pytorch/extension-cpp/, Issue #71
# TODO: a proper setup script to install a compiled version
# TODO: check whether the module has already been installed. If so, then skip
from torch.utils.cpp_extension import load

mc = load(
    name="radius_kernel",
    sources=[path.join(path.dirname(__file__), "radius.cu")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

def radius_distance_fn(
    x: torch.Tensor,
    r: float,
    batch: Optional[torch.Tensor] = None,
    loop: bool = False,
    max_num_neighbors: int = 32,
    batch_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if x.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long, device=x.device)

    if batch_size is None:
        batch_size = 1
        if batch is not None:
            assert x.size(0) == batch.numel()
            batch_size = int(batch.max()) + 1
    assert batch_size > 0
    ptr_x: Optional[torch.Tensor] = None

    if batch_size > 1:
        assert batch is not None
        arange = torch.arange(batch_size + 1, device=x.device)
        ptr_x = torch.bucketize(arange, batch)

    return mc.radius_cuda(x, ptr_x, r,
                          max_num_neighbors,
                          not loop)

class RadiusDistanceAGF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, r, batch, loop, max_num_neighbors, batch_size):
        edge_index, distance = radius_distance_fn(x, r, batch, loop, 
                                                  max_num_neighbors, 
                                                  batch_size)
        #distance.requires_grad_()
        ctx.save_for_backward(x, distance, edge_index)
        ctx.mark_non_differentiable(edge_index)
        return distance, edge_index 

    @staticmethod
    def backward(ctx, grad_d, grad_ei):
        x, distance, edge_index = ctx.saved_tensors
        return RadiusDistanceAGB.apply(x, distance, grad_d, edge_index)

class RadiusDistanceAGB(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, distance, grad_d, edge_index):
        edge_i = edge_index[0]
        edge_j = edge_index[1]
        differences = x[edge_i] - x[edge_j]
        scaling = (grad_d / distance).unsqueeze(-1)
        contrib_i = scaling * differences
        contrib_j = -scaling * differences
        grad_x = torch.zeros_like(x)
        grad_x.index_add_(0, edge_i, contrib_i)
        grad_x.index_add_(0, edge_j, contrib_j)

        ctx.save_for_backward(edge_index, differences, scaling, grad_d, distance)
        return grad_x, None, None, None, None, None

    @staticmethod
    def backward(ctx, grad_out_x, *args):
        edge_index, differences, scaling, grad_d, distance = ctx.saved_tensors
        edge_i, edge_j = edge_index[0], edge_index[1]

        delta_grad = grad_out_x[edge_i] - grad_out_x[edge_j]
        delta_grad_dot_diff = (delta_grad * differences).sum(dim=1)

        term = scaling * delta_grad
        grad_x = torch.zeros_like(grad_out_x)
        grad_x.index_add_(0, edge_i, term)
        grad_x.index_add_(0, edge_j, -term)

        grad_distance = -grad_d * delta_grad_dot_diff / (distance ** 2)

        grad_grad_d = delta_grad_dot_diff / distance

        return grad_x, grad_distance, grad_grad_d, None
    #@staticmethod
    #def forward(ctx, x, distance, grad_d, edge_index):
    #    e_i, e_j = edge_index[0], edge_index[1]
    #    diff = x[e_i] - x[e_j]
    #    tmp  = (grad_d / distance).unsqueeze(-1)
    #    part = tmp * diff
    #    grad_x = torch.zeros_like(x)
    #    grad_x.index_add_(0, e_i,  part)
    #    grad_x.index_add_(0, e_j, -part)
    #    ctx.save_for_backward(edge_index, diff, tmp, grad_d, distance)
    #    return grad_x, None, None, None, None, None

    #@staticmethod
    #def backward(ctx, grad_out_x, *args):
    #    edge_index, diff, part, grad_d, distance = ctx.saved_tensors
    #    e_i, e_j = edge_index[0], edge_index[1]
    #    delta_grad = grad_out_x[e_i] - grad_out_x[e_j]
    #    delta_grad_dot_diff = (delta_grad * diff).sum(dim=1)
    #    term = part * delta_grad
    #    grad_x = torch.zeros_like(grad_out_x)
    #    grad_x.index_add_(0, e_i,  term)
    #    grad_x.index_add_(0, e_j, -term)
    #    grad_distance = -grad_d * delta_grad_dot_diff / (distance ** 2)
    #    grad_grad_d = delta_grad_dot_diff / distance
    #    return grad_x, grad_distance, grad_grad_d, None

def radius_distance(
    x: torch.Tensor,
    r: float,
    batch: Optional[torch.Tensor] = None,
    loop: bool = False,
    max_num_neighbors: int = 32,
    batch_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return RadiusDistanceAGF.apply(x, r, batch, loop, max_num_neighbors, batch_size)
