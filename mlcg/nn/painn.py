"""Code adapted from https://github.com/atomistic-machine-learning/schnetpack"""

from typing import Callable, Dict, Optional, Union, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
import warnings

from mlcg.data import AtomicData, ENERGY_KEY
from mlcg.nn import MLP, Dense
from mlcg.geometry.internal_coordinates import compute_distances
from mlcg.neighbor_list.neighbor_list import (
    atomic_data2neighbor_list,
    validate_neighborlist,
)



class PaiNNInteraction(MessagePassing):
    r"""torch_geometric implementation of PaiNN block for modeling equivariant interactions. 
    Parameters
    -----------
    hidden_channels: 
        hidden channel dimension, i.e. node feature size used for the node embedding 
    edge_attr_dim:
        edge attributes dimension, i.e. number of radial basis functions 
    cutoff: 
        cutoff function
    activation: Callable, 
    aggr: str = "add"
    """
    
    def __init__(
            self, 
            hidden_channels: int, 
            edge_attr_dim: int,
            cutoff: nn.Module,
            activation: Callable, 
            aggr: str = "add"):
        super().__init__(aggr=aggr)
        self.hidden_channels = hidden_channels

        self.interatomic_context_net = nn.Sequential(
            Dense(hidden_channels, hidden_channels, activation=activation),
            Dense(hidden_channels, 3 * hidden_channels, activation=None),
        )
        self.filter_network = Dense(edge_attr_dim, 3*hidden_channels, activation=None)
        self.cutoff = cutoff

    def reset_parameters(self):
        pass #FIXME:change accordingly to DENSE

    def forward(
            self,
            scalar_node_features: torch.Tensor, # (n_nodes, 1, n_feat)
            vector_node_features: torch.Tensor, # (n_nodes, 3, n_feat)
            normdir: torch.Tensor, # (n_edges, 3)
            edge_index: torch.Tensor, # (2, n_edges)
            edge_weight: torch.Tensor, # (n_edges)
            edge_attr: torch.Tensor # (n_edges, edge_attr_dim)
    ):
        """Compute interaction output.

        Args:
            scalar_node_features: 
                scalar input values per node with shape (n_nodes, 1, n_feat)
            vector_node_features: 
                vector input values per node with shape (n_nodes, 3, n_feat)
            normdir: 
                normalized directions for every edge with shape (total_num_edges, 3)
            edge_index: 
                graph edge index tensor of shape (2, total_num_edges)
            edge_weight: 
                scalar edge weight, i.e. distances, of shape (n_edges, 1)
            edge_attr
                edge attributes, i.e. rbf projection, of shape (n_edges, edge_attr_dim)

        Returns:
            scalar and vector features after interaction
        """
        C = self.cutoff(edge_weight)
        W = self.filter_network(edge_attr) * C.view(-1, 1)

        x_scalar = scalar_node_features.squeeze(1) # (n_nodes, n_feat)
        n_nodes, _ = x_scalar.shape
        x_vector = vector_node_features.view(n_nodes, -1) # (n_nodes, 3*n_feat)
        x = self.interatomic_context_net(x_scalar)
        return self.propagate(edge_index, x=x, x_scalar=x_scalar, x_vector=x_vector, W=W, normdir=normdir)
    
    def message(self, x_j, x_vector_j, W, normdir):
        x_j = x_j.unsqueeze(1) # reshape as (n_nodes, 1, 3*n_feats)
        x_vector_j = x_vector_j.view(x_j.shape[0], 3, -1)  # reshape as (n_nodes, 3, n_feats)
        x = W * x_j
        dq, dmuR, dmumu = torch.split(x, self.hidden_channels, dim=-1)
        dmu = dmuR * normdir.unsqueeze(-1) + dmumu * x_vector_j
        return dq, dmu

    def aggregate(self, inputs, index, dim_size):
        dq, dmu = inputs
        dq = scatter(dq, index, dim=0, dim_size=dim_size)
        dmu = scatter(dmu, index, dim=0, dim_size=dim_size)
        return dq, dmu

    def update(self, inputs, x_scalar, x_vector):
        dq, dmu = inputs
        x_scalar = x_scalar.unsqueeze(1) + dq
        x_vector = x_vector.view(x_scalar.shape[0], 3, -1) + dmu
        return x_scalar, x_vector


class PaiNNMixing(nn.Module):
    r"""PaiNN interaction block for mixing on atom features."""

    def __init__(self, hidden_channels: int, activation: Callable, epsilon: float = 1e-8):
        """
        Args:
            hidden_channels: number of features to describe atomic environments.
            activation: if None, no activation function is used.
            epsilon: stability constant added in norm to prevent numerical instabilities
        """
        super(PaiNNMixing, self).__init__()
        self.hidden_channels = hidden_channels

        self.intraatomic_context_net = nn.Sequential(
            Dense(2 * hidden_channels, hidden_channels, activation=activation),
            Dense(hidden_channels, 3 * hidden_channels, activation=None),
        )
        self.mu_channel_mix = Dense(
            hidden_channels, 2 * hidden_channels, activation=None, bias=False
        )
        self.epsilon = epsilon

    def forward(self, q: torch.Tensor, mu: torch.Tensor):
        """Compute intraatomic mixing.

        Args:
            q: scalar input values
            mu: vector input values

        Returns:
            atom features after interaction
        """
        ## intra-atomic
        mu_mix = self.mu_channel_mix(mu)
        mu_V, mu_W = torch.split(mu_mix, self.hidden_channels, dim=-1)
        mu_Vn = torch.sqrt(torch.sum(mu_V**2, dim=-2, keepdim=True) + self.epsilon)

        ctx = torch.cat([q, mu_Vn], dim=-1)
        x = self.intraatomic_context_net(ctx)
        
        # FIXME: check 
        dq_intra, dmu_intra, dqmu_intra = torch.split(x, self.hidden_channels, dim=-1)
        dmu_intra = dmu_intra * mu_W

        dqmu_intra = dqmu_intra * torch.sum(mu_V * mu_W, dim=1, keepdim=True)

        q = q + dq_intra + dqmu_intra
        mu = mu + dmu_intra
        return q, mu

class PaiNN(nn.Module):
    """PaiNN - polarizable interaction neural network

    References:

    .. [#painn1] Schütt, Unke, Gastegger:
       Equivariant message passing for the prediction of tensorial properties and molecular spectra.
       ICML 2021, http://proceedings.mlr.press/v139/schutt21a.html

    """

    def __init__(
        self,
        embedding_layer: nn.Module,
        interaction_blocks: Union[PaiNNInteraction, List[PaiNNInteraction]],
        mixing_blocks: Union[PaiNNMixing, List[PaiNNMixing]],
        rbf_layer: torch.nn.Module,
        output_network: nn.Module,
        max_num_neighbors: int,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            n_interactions: number of interaction blocks.
            radial_basis: layer for expanding interatomic distances in a basis set
            cutoff_fn: cutoff function
            activation: activation function
            shared_interactions: if True, share the weights across
                interaction blocks.
            shared_interactions: if True, share the weights across
                filter-generating networks.
            epsilon: numerical stability parameter
            nuclear_embedding: custom nuclear embedding (e.g. spk.nn.embeddings.NuclearEmbedding)
            electronic_embeddings: list of electronic embeddings. E.g. for spin and
                charge (see spk.nn.embeddings.ElectronicEmbedding)
        """
        super(PaiNN, self).__init__()

        self.embedding_layer = embedding_layer
        if isinstance(interaction_blocks, List):
            self.interaction_blocks = torch.nn.ModuleList(*interaction_blocks)
        elif isinstance(interaction_blocks, PaiNNInteraction):
            self.interaction_blocks = torch.nn.Sequential(interaction_blocks)
        else:
            raise RuntimeError(
                "interaction_blocks must be a single InteractionBlock or "
                "a list of InteractionBlocks."
            )
        if isinstance(mixing_blocks, List):
            self.mixing_blocks = torch.nn.ModuleList(*mixing_blocks)
        elif isinstance(mixing_blocks, PaiNNMixing):
            self.mixing_blocks = torch.nn.Sequential(mixing_blocks)
        else:
            raise RuntimeError(
                "mixing_blocks must be a single PaiNNMixing or "
                "a list of PaiNNMixing."
            )
        self.output_network = output_network
        self.rbf_layer = rbf_layer
        self.max_num_neighbors = max_num_neighbors

        self.reset_parameters()

    def reset_parameters(self):
        """Method for resetting linear layers in each SchNet component"""
        self.embedding_layer.reset_parameters()
        self.rbf_layer.reset_parameters()
        for block in self.interaction_blocks:
            block.reset_parameters()
        for block in self.mixing_blocks:
            block.reset_parameters()
        self.output_network.reset_parameters()

    # def forward(self, inputs: Dict[str, torch.Tensor]):
    def forward(self, data:AtomicData):
        """
        Compute atomic representations/embeddings.

        Args:
            inputs: SchNetPack dictionary of input tensors.

        Returns:
            torch.Tensor: atom-wise representation.
            list of torch.Tensor: intermediate atom-wise representations, if
            return_intermediate=True was used.
        """
        neighbor_list = data.neighbor_list.get(self.name)

        if not self.is_nl_compatible(neighbor_list):
            neighbor_list = self.neighbor_list(
                data, self.rbf_layer.cutoff.cutoff_upper, self.max_num_neighbors
            )[self.name]
        edge_index = neighbor_list["index_mapping"]
        distances = compute_distances(
            data.pos,
            edge_index,
            neighbor_list["cell_shifts"],
        )
        normdir = (data.pos[edge_index[1]]-data.pos[edge_index[0]]) / distances
        rbf_expansion = self.rbf_layer(distances)
        num_batch = data.batch[-1] + 1

        q = self.embedding_layer(data.atom_types) # (n_atoms, n_features)
        q = q.unsqueeze(1) # (n_atoms, 1, n_features)
        mu = torch.zeros((q.shape[0], 3, q.shape[2]), device=q.device) # (n_atoms, 3, n_features)

        for i, (interaction, mixing) in enumerate(zip(self.interaction_blocks, self.mixing_blocks)):
            q, mu = interaction(q, mu, normdir, edge_index, distances, rbf_expansion)
            q, mu = mixing(q, mu)
        q = q.squeeze(1)

        energy = self.output_network(q, data)
        energy = scatter(energy, data.batch, dim=0, reduce="sum")
        energy = energy.flatten()
        data.out[self.name] = {ENERGY_KEY: energy}

        return data

    def is_nl_compatible(self, nl):
        is_compatible = False
        if validate_neighborlist(nl):
            if (
                nl["order"] == 2
                and nl["self_interaction"] == False
                and nl["rcut"] == self.cutoff.cutoff_upper
            ):
                is_compatible = True
        return is_compatible

class StandardPaiNN(PaiNN):
    """mall wrapper class for :ref:`PaiNN` to simplify the definition of the
    PaiNN model through an input file. The upper distance cutoff attribute
    in is set by default to match the upper cutoff value in the cutoff function."""
    def __init__(
        self,
        rbf_layer: torch.nn.Module,
        cutoff: torch.nn.Module,
        output_hidden_layer_widths: List[int],
        hidden_channels: int = 128,
        embedding_size: int = 100,
        num_interactions: int = 3,
        activation: torch.nn.Module = torch.nn.Tanh(),
        max_num_neighbors: int = 1000,
        aggr: str = "add",
        epsilon: float = 1e-8,
    ):
        if num_interactions < 1:
            raise ValueError("At least one interaction block must be specified")

        if cutoff.cutoff_lower != rbf_layer.cutoff.cutoff_lower:
            warnings.warn(
                "Cutoff function lower cutoff, {}, and radial basis function "
                " lower cutoff, {}, do not match.".format(
                    cutoff.cutoff_lower, rbf_layer.cutoff.cutoff_lower
                )
            )
        if cutoff.cutoff_upper != rbf_layer.cutoff.cutoff_upper:
            warnings.warn(
                "Cutoff function upper cutoff, {}, and radial basis function "
                " upper cutoff, {}, do not match.".format(
                    cutoff.cutoff_upper, rbf_layer.cutoff.cutoff_upper
                )
            )

        embedding_layer = torch.nn.Embedding(embedding_size, hidden_channels)

        interaction_blocks = []
        mixing_blocks = []
        for _ in range(num_interactions):
            interaction_blocks.appned(
                PaiNNInteraction(
                    hidden_channels,
                    rbf_layer.num_rbf,
                    cutoff,
                    activation,
                    aggr
                )
            )
            mixing_blocks.append(
                PaiNNMixing(
                    hidden_channels,
                    activation,
                    epsilon
                )
            )

        output_layer_widths = (
            [hidden_channels] + output_hidden_layer_widths + [1]
        )
        output_network = MLP(
            output_layer_widths, activation_func=activation, last_bias=False
        )
        super(PaiNN, self).__init__(
            embedding_layer,
            interaction_blocks,
            mixing_blocks,
            rbf_layer,
            output_network,
            max_num_neighbors,
        )