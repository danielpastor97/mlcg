from typing import Optional, NamedTuple
import torch
from .torch_impl import torch_neighbor_list


class NeighborList(NamedTuple):
    """data structure holding the information about connectivity within atomic
    structures.
    """

    #: quick identifier for compatibility checking
    tag: Optional[str] = None
    #: an int providing the order of the neighborlist, e.g. order == 2 means that
    #: central atoms `i` have 1 neighbor `j` so distances can be computed,
    #: order == 3 means that central atoms `i` have 2 neighbors `j` and `k` so
    #: angles can be computed
    order: Optional[int] = None
    #: The [2, n_edge] index tensor giving center -> neighbor relations. 1st column
    #: refers to the central atom index and the 2nd column to the neighbor atom
    #: index in the list of atoms (so it has to be shifted by a cell_shift to get
    #: the actual position of the neighboring atoms)
    index_mapping: Optional[torch.Tensor] = None
    #: A [n_edge, 3] tensor giving the periodic cell shift
    cell_shifts: Optional[torch.Tensor] = None
    #: cutoff radius used to compute the connectivity
    rcut: Optional[float] = None
    #: wether the mapping includes self refferring mappings, e.g. mappings where
    #: `i` == `j`.
    self_interaction: Optional[bool] = None

    @staticmethod
    def from_topology(topology, type: str = "bonds"):
        """Build Neighborlist from a :ref:`mlcg.neighbor_list.neighbor_list.Topology`.

        Parameters
        ----------
        topology: :ref:`mlcg.neighbor_list.neighbor_list.Topology`
            A topology object.
        type:
            kind of information to extract (should be in ["bonds", "angles",
            "dihedrals"]).
        """
        assert type in ["bonds", "angles", "dihedrals"]
        if type == "bonds":
            nl = NeighborList(
                tag=type,
                order=2,
                mapping=topology.bonds2torch(),
                self_interaction=False,
            )
        elif type == "angles":
            nl = NeighborList(
                tag=type,
                order=3,
                mapping=topology.angles2torch(),
                self_interaction=False,
            )
        elif type == "dihedrals":
            nl = NeighborList(
                tag=type,
                order=4,
                mapping=topology.dihedrals2torch(),
                self_interaction=False,
            )

        return nl

    @staticmethod
    def from_atomic_data(
        data,
        rcut: float,
        self_interaction: bool = False,
    ):

        idx_i, idx_j, cell_shifts, _ = torch_neighbor_list(
            data, rcut, self_interaction=self_interaction
        )

        mapping = torch.cat([idx_i.unsqueeze(0), idx_j.unsqueeze(0)], dim=0)
        order = mapping.shape[0]
        return NeighborList(
            tag=f"unbounded rc:{rcut} order:{order}",
            order=order,
            index_mapping=mapping,
            cell_shifts=cell_shifts,
            rcut=rcut,
            self_interaction=self_interaction,
        )
