from typing import Any, Callable, Dict, List, Optional, Union, Final

import torch
from e3nn import o3

try:
    from mace.modules.radial import ZBLBasis
    from mace.tools.scatter import scatter_sum
    from mace.tools import to_one_hot

    from mace.modules.blocks import (
        EquivariantProductBasisBlock,
        LinearNodeEmbeddingBlock,
        LinearReadoutBlock,
        NonLinearReadoutBlock,
        RadialEmbeddingBlock,
    )
    from mace.modules.utils import get_edge_vectors_and_lengths

except ImportError as e:
    print(e)
    print("Please install or set mace to your path before using this interface. " +
          "To install you can either run 'pip install git+https://github.com/ACEsuit/mace.git', " +
          "or clone the repository and add it to your PYTHONPATH.""")

from mlcg.pl.model import get_class_from_str
from mlcg.data.atomic_data import AtomicData, ENERGY_KEY
from mlcg.neighbor_list.neighbor_list import (
    atomic_data2neighbor_list,
    validate_neighborlist,
)

from e3nn.util.jit import compile_mode


@compile_mode("script")
class MACE(torch.nn.Module):
    name: Final[str] = "MACE"

    def __init__(
        self,
        atomic_numbers: torch.Tensor,
        node_embedding: torch.nn.Module,
        radial_embedding: torch.nn.Module,
        spherical_harmonics: torch.nn.Module,
        interactions: List[torch.nn.Module],
        products: List[torch.nn.Module],
        readouts: List[torch.nn.Module],
        r_max: float,
        max_num_neighbors: int,
        pair_repulsion_fn: torch.nn.Module = None,
    ):
        super().__init__()

        self.register_buffer(
                "atomic_numbers", atomic_numbers
            )
        self.node_embedding = node_embedding
        self.radial_embedding = radial_embedding
        self.spherical_harmonics = spherical_harmonics
        self.interactions = torch.nn.ModuleList(interactions)
        self.products = torch.nn.ModuleList(products)
        self.readouts = torch.nn.ModuleList(readouts)
        self.r_max = r_max
        self.max_num_neighbors = max_num_neighbors
        self.pair_repulsion_fn = pair_repulsion_fn

        self.register_buffer(
                "types_mapping",
                -1 * torch.ones(atomic_numbers.max() + 1, dtype=torch.long),
            )
        self.types_mapping[atomic_numbers] = torch.arange(atomic_numbers.shape[0])

    def forward(self, data: AtomicData) -> AtomicData:
        # Setup
        num_atoms_arange = torch.arange(data.pos.shape[0])
        num_graphs = data.ptr.numel() - 1  # data.batch.max()
        node_heads = torch.zeros_like(data.batch)

        types_ids = self.types_mapping[data.atom_types].view(-1, 1)
        node_attrs = to_one_hot(types_ids, self.atomic_numbers.shape[0])

        # Embeddings
        node_feats = self.node_embedding(node_attrs)

        neighbor_list = data.neighbor_list.get(self.name)

        if not self.is_nl_compatible(neighbor_list):
            neighbor_list = self.neighbor_list(
                data, self.r_max, self.max_num_neighbors
            )[self.name]

        edge_index = neighbor_list["index_mapping"]

        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data.pos,
            edge_index=edge_index,
            shifts=neighbor_list["cell_shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(
            lengths, node_attrs, edge_index, self.atomic_numbers
        )

        if self.pair_repulsion_fn:
            pair_node_energy = self.pair_repulsion_fn(
                lengths, node_attrs, edge_index, self.atomic_numbers
            )
            pair_energy = scatter_sum(
                src=pair_node_energy, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
        else:
            pair_energy = torch.zeros(data.batch.max() + 1,
                                      device=data.pos.device,
                                      dtype=data.pos.dtype)

        # Interactions
        energies = [pair_energy]
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
            )
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=node_attrs
            )
            node_energies = readout(node_feats, node_heads)[
                num_atoms_arange, node_heads
            ]  # [n_nodes, len(heads)]
            energy = scatter_sum(
                src=node_energies,
                index=data["batch"],
                dim=0,
                dim_size=num_graphs,
            )  # [n_graphs,]
            energies.append(energy)

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]

        data.out[self.name] = {ENERGY_KEY: total_energy}

        return data

    def is_nl_compatible(self, nl):
        is_compatible = False
        if validate_neighborlist(nl):
            if (
                nl["order"] == 2
                and nl["self_interaction"] is False
                and nl["rcut"] == self.r_max
            ):
                is_compatible = True
        return is_compatible

    @staticmethod
    def neighbor_list(
        data: AtomicData, rcut: float, max_num_neighbors: int = 1000
    ) -> dict:
        """Computes the neighborlist for :obj:`data` using a strict cutoff of :obj:`rcut`."""
        return {
            MACE.name: atomic_data2neighbor_list(
                data,
                rcut,
                self_interaction=False,
                max_num_neighbors=max_num_neighbors,
            )
        }


@compile_mode("script")
class StandardMACE(MACE):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: str,
        interaction_cls_first: str,
        num_interactions: int,
        hidden_irreps: str,
        MLP_irreps: str,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: Union[int, List[int]],
        gate: Optional[Callable],
        max_num_neighbors: int = 1000,
        pair_repulsion: bool = False,
        distance_transform: str = "None",
        radial_MLP: Optional[List[int]] = None,
        radial_type: Optional[str] = "bessel",
        cueq_config: Optional[Dict[str, Any]] = None,
    ):
        atomic_numbers.sort()
        atomic_numbers = torch.as_tensor(atomic_numbers)
        num_elements = atomic_numbers.shape[0]

        hidden_irreps = o3.Irreps(hidden_irreps)
        MLP_irreps = o3.Irreps(MLP_irreps)

        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps,
            irreps_out=node_feats_irreps,
            cueq_config=cueq_config,
        )
        radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
            distance_transform=distance_transform,
        )
        edge_feats_irreps = o3.Irreps(f"{radial_embedding.out_dim}x0e")

        pair_repulsion_fn = None
        if pair_repulsion:
            pair_repulsion_fn = ZBLBasis(p=num_polynomial_cutoff)

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]

        # Interactions and readout
        inter = get_class_from_str(interaction_cls_first)(
                    node_attrs_irreps=node_attr_irreps,
                    node_feats_irreps=node_feats_irreps,
                    edge_attrs_irreps=sh_irreps,
                    edge_feats_irreps=edge_feats_irreps,
                    target_irreps=interaction_irreps,
                    hidden_irreps=hidden_irreps,
                    avg_num_neighbors=avg_num_neighbors,
                    radial_MLP=radial_MLP,
                    cueq_config=cueq_config
                    )
        interactions = [inter]

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in interaction_cls_first:
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation[0],
            num_elements=num_elements,
            use_sc=use_sc_first,
            cueq_config=cueq_config,
        )
        products = [prod]

        readouts = [
            LinearReadoutBlock(
                hidden_irreps, o3.Irreps("1x0e"), cueq_config
                )
            ]

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                hidden_irreps_out = str(
                    hidden_irreps[0]
                )  # Select only scalars for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = get_class_from_str(interaction_cls)(
                        node_attrs_irreps=node_attr_irreps,
                        node_feats_irreps=hidden_irreps,
                        edge_attrs_irreps=sh_irreps,
                        edge_feats_irreps=edge_feats_irreps,
                        target_irreps=interaction_irreps,
                        hidden_irreps=hidden_irreps_out,
                        avg_num_neighbors=avg_num_neighbors,
                        radial_MLP=radial_MLP,
                        cueq_config=cueq_config
                        )
            interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i + 1],
                num_elements=num_elements,
                use_sc=True,
                cueq_config=cueq_config,
            )
            products.append(prod)
            if i == num_interactions - 2:
                readouts.append(
                    NonLinearReadoutBlock(
                        hidden_irreps_out,
                        (1 * MLP_irreps).simplify(),
                        gate,
                        o3.Irreps("1x0e"),
                        1,
                        cueq_config,
                    )
                )
            else:
                readouts.append(
                    LinearReadoutBlock(
                        hidden_irreps, o3.Irreps("1x0e"), cueq_config
                    )
                )

        super().__init__(
            atomic_numbers,
            node_embedding,
            radial_embedding,
            spherical_harmonics,
            interactions,
            products,
            readouts,
            r_max,
            max_num_neighbors,
            pair_repulsion_fn,
        )
