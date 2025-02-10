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
    indxs = torch.arange(data.atom_types.size()[0],device=data.pos.device)
    indxs_roll_front = indxs.roll(1)
    batch_roll_front = data.batch.roll(1)
    indxs_roll_back = indxs.roll(-1)
    batch_roll_back = data.batch.roll(-1)
    # we get separetely the neighbors to one side and then to the other side
    front_neighs = torch.vstack((indxs, indxs_roll_front))[
        :, batch_roll_front == data.batch
    ]
    back_neighs = torch.vstack((indxs, indxs_roll_back))[
        :, batch_roll_back == data.batch
    ]
    total_neighs = torch.hstack((front_neighs, back_neighs))
    return total_neighs
