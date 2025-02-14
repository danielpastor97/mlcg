import torch
from ..data.atomic_data import AtomicData


def get_seq_neigh(data: AtomicData) -> torch.Tensor:
    r"""Build neighbor list of sequence neighbors in the atom_types

    This takes into account data that is in different batches.

    Parameters
    ----------

    data:
        AtomicData instance from which we want the neighboring sequence

    Returns
    -------

    total_neighs:
        A torch tensor of shape (2,n_neighs)


    Notes:
    -------
        This function returns neighbors in the order of the atom types

    """
    indxs = torch.arange(data.atom_types.size()[0], device=data.pos.device)
    indxs_shift_front = indxs[1:]
    batch_shift_front = data.batch[1:]
    indxs_shift_back = indxs[:-1]
    batch_shift_back = data.batch[:-1]
    mask = (
        batch_shift_front == batch_shift_back
    )  # mask pairs not from the same frame
    # we get separetely the neighbors to one side and then to the other side
    front_neighs = torch.vstack((indxs_shift_front, indxs_shift_back))[:, mask]
    back_neighs = torch.vstack((indxs_shift_back, indxs_shift_front))[:, mask]
    total_neighs = torch.hstack((front_neighs, back_neighs))
    return total_neighs
