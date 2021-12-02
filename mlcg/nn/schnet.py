import warnings
from typing import Optional, List, Final
import torch
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from ..neighbor_list.neighbor_list import (
    atomic_data2neighbor_list,
    validate_neighborlist,
)
from ..data.atomic_data import AtomicData, ENERGY_KEY
from ..geometry.internal_coordinates import compute_distances
from .mlp import MLP
from ._module_init import init_xavier_uniform


class SchNet(torch.nn.Module):
    r"""PyTorch Geometric implementation of SchNet
    Code adapted from [PT_geom_schnet]_  which is based on the architecture
    described in [Schnet]_ .

    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/schnet.html

    Parameters
    ----------
    embedding_layer:
        Initial embedding layer that transforms atoms/coarse grain bead
        types into embedded features
    interaction_blocks: list of torch.nn.Module or torch.nn.Sequential
        Sequential interaction blocks of the model, where each interaction
        block applies
    rbf_layer:
        The set of radial basis functions that expands pairwise distances
        between atoms/CG beads.
    output_network:
        Output neural network that predicts scalar energies from SchNet
        features. This network should transform (num_examples * num_atoms,
        hidden_channels) to (num_examples * num atoms, 1).
    upper_distance_cutoff:
        Upper distance cutoff used for making neighbor lists.
    self_interaction:
        If True, self interactions/distancess are calculated.
    max_num_neighbors:
        Maximum number of neighbors to return for a
        given node/atom when constructing the molecular graph during forward
        passes. This attribute is passed to the torch_cluster radius_graph
        routine keyword max_num_neighbors, which normally defaults to 32.
        Users should set this to higher values if they are using higher upper
        distance cutoffs and expect more than 32 neighbors per node/atom.
    """

    name: Final[str] = "SchNet"

    def __init__(
        self,
        embedding_layer: torch.nn.Module,
        interaction_blocks: List[torch.nn.Module],
        rbf_layer: torch.nn.Module,
        output_network: torch.nn.Module,
        self_interaction: bool = False,
        max_num_neighbors: int = 1000,
    ):

        super(SchNet, self).__init__()

        self.embedding_layer = embedding_layer
        self.rbf_layer = rbf_layer
        self.max_num_neighbors = max_num_neighbors
        self.self_interaction = self_interaction

        if isinstance(interaction_blocks, List):
            self.interaction_blocks = torch.nn.Sequential(*interaction_blocks)
        elif isinstance(interaction_blocks, InteractionBlock):
            self.interaction_blocks = torch.nn.Sequential(interaction_blocks)
        else:
            raise RuntimeError(
                "interaction_blocks must be a single InteractionBlock or "
                "a list of InteractionBlocks."
            )

        self.output_network = output_network
        self.reset_parameters()

    def reset_parameters(self):
        """Method for resetting linear layers in each SchNet component"""
        self.embedding_layer.reset_parameters()
        self.rbf_layer.reset_parameters()
        for block in self.interaction_blocks:
            block.reset_parameters()
        self.output_network.reset_parameters()

    def forward(self, data: AtomicData) -> AtomicData:
        r"""Forward pass through the SchNet architecture.

        Parameters
        ----------
        data:
            Input data object containing batch atom/bead positions
            and atom/bead types.

        Returns
        -------
        data:
           Data dictionary, updated with predicted energy of shape
           (num_examples * num_atoms, 1), as well as neighbor list
           information.
        """
        x = self.embedding_layer(data.atom_types)

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

        rbf_expansion = self.rbf_layer(distances)

        for block in self.interaction_blocks:
            x = x + block(x, edge_index, distances, rbf_expansion)

        energy = self.output_network(x)
        energy = scatter(energy, data.batch, dim=0, reduce="sum")

        energy = energy.squeeze()
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

    @staticmethod
    def neighbor_list(
        data: AtomicData, rcut: float, max_num_neighbors: int = 1000
    ) -> dict:
        """Computes the neighborlist for :obj:`data` using a strict cutoff of :obj:`rcut`."""
        return {
            SchNet.name: atomic_data2neighbor_list(
                data,
                rcut,
                self_interaction=False,
                max_num_neighbors=max_num_neighbors,
            )
        }


class InteractionBlock(torch.nn.Module):
    r"""Interaction blocks for SchNet. Consists of atomwise
    transformations of embedded features that are continuously
    convolved with filters generated from radial basis function-expanded
    pairwise distances.

    Parameters
    ----------
    cfconv_layer:
        Continuous filter convolution layer for convolutions of radial basis
        function-expanded distances with embedded features
    hidden_channels:
        Hidden dimension of embedded features
    activation:
        Activation function applied to linear layer outputs
    """

    def __init__(
        self,
        cfconv_layer: torch.nn.Module,
        hidden_channels: int = 128,
        activation: type = torch.nn.Tanh,
    ):
        super(InteractionBlock, self).__init__()
        self.conv = cfconv_layer
        self.activation = activation
        self.lin = torch.nn.Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        init_xavier_uniform(self.lin)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        r"""Forward pass through the interaction block.

        Parameters
        ----------
        x:
            Embedded features of shape (num_examples, num_atoms,
            hidden_channels)
        edge_index:
            Graph edge index tensor of shape (2, total_num_edges)
        edge_weight:
            Graph edge weight (eg, distances), of shape (total_num_edges)
        edge_attr:
            Graph edge attributes (eg, expanded distances), of shape
            (total_num_edges, num_rbf)

        Returns
        -------
        x:
            Updated embedded features of shape (num_examples * num_atoms,
            hidden_channels)
        """

        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.activation(x)
        x = self.lin(x)
        return x


class CFConv(MessagePassing):
    r"""Continuous filter convolutions for `SchNet`.

    Parameters
    ----------
    filter_net:
        Neural network for generating filters from expanded pairwise distances
    cutoff:
        Cutoff envelope to apply to the output of the filter generating network.
    in_channels:
        Hidden input dimensions
    out_channels:
        Hidden output dimensions
    num_filters:
        Number of filters
    aggr:
        Aggregation scheme for continuous filter output. For all options,
        see https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html?highlight=MessagePassing#the-messagepassing-base-class
    """

    def __init__(
        self,
        filter_network: torch.nn.Module,
        cutoff: torch.nn.Module,
        in_channels: int = 128,
        out_channels: int = 128,
        num_filters: int = 128,
        aggr: str = "add",
    ):
        super(CFConv, self).__init__(aggr=aggr)
        self.lin1 = torch.nn.Linear(in_channels, num_filters, bias=False)
        self.lin2 = torch.nn.Linear(num_filters, out_channels)
        self.filter_network = filter_network
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        r"""Method for resetting the weights of the linear
        layers and filter network according the the
        Xavier uniform strategy. Biases
        are set to 0.
        """

        self.filter_network.reset_parameters()
        init_xavier_uniform(self.lin1)
        init_xavier_uniform(self.lin2)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        r"""Forward pass through the continuous filter convolution.

        Parameters
        ----------
        x:
            Embedded features of shape (num_examples * num_atoms,
            hidden_channels)
        edge_index
            Graph edge index tensor of shape (2, total_num_edges)
        edge_weight:
            Graph edge weight (eg, distances), of shape (total_num_edges)
        edge_attr:
            Graph edge attributes (eg, expanded distances), of shape
            (total_num_edges, num_rbf)

        Returns
        -------
        x:
            Updated embedded features of shape (num_examples * num_atoms,
            hidden_channels)
        """
        C = self.cutoff(edge_weight)
        W = self.filter_network(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        # propagate_type: (x: Tensor, W: Tensor)
        # Perform the continuous filter convolution
        x = self.propagate(edge_index, x=x, W=W, size=None)
        x = self.lin2(x)
        return x

    def message(self, x_j: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        r"""Message passing operation to perform the continuous filter
        convolution through element-wise multiplcation of embedded
        features with the output of the filter network.

        Parameters
        ----------
        x_j:
            Tensor of embedded features of shape (total_num_edges,
            hidden_channels)
        W:
            Tensor of filter values of shape (total_num_edges, num_filters)

        Returns
        -------
        x_j * W:
            Elementwise multiplication of the filters with embedded features.
        """
        return x_j * W


class StandardSchNet(SchNet):
    """Small wrapper class for :ref:`SchNet` to simplify the definition of the
    SchNet model through an input file. The upper distance cutoff attribute
    in is set by default to match the upper cutoff value in the cutoff function.

    Parameters
    ----------
    rbf_layer:
        radial basis function used to project the distances :math:`r_{ij}`.
    cutoff:
        smooth cutoff function to supply to the CFConv
    output_hidden_layer_widths:
        List giving the number of hidden nodes of each hidden layer of the MLP
        used to predict the target property from the learned representation.
    hidden_channels:
        dimension of the learned representation, i.e. dimension of the embeding projection, convolution layers, and interaction block.
    embedding_size:
        dimension of the input embeddings (should be larger than :obj:`AtomicData.atom_types.max()+1`).
    num_filters:
        number of nodes of the networks used to filter the projected distances
    num_interactions:
        number of interaction blocks
    activation:
        activation function
    max_num_neighbors:
        The maximum number of neighbors to return for each atom in :obj:`data`.
        If the number of actual neighbors is greater than
        :obj:`max_num_neighbors`, returned neighbors are picked randomly.
    aggr:
        Aggregation scheme for continuous filter output. For all options,
        see
         `aggr <https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html?highlight=MessagePassing#the-messagepassing-base-class>`_

    """

    def __init__(
        self,
        rbf_layer: torch.nn.Module,
        cutoff: torch.nn.Module,
        output_hidden_layer_widths: List[int],
        hidden_channels: int = 128,
        embedding_size: int = 100,
        num_filters: int = 128,
        num_interactions: int = 3,
        activation: torch.nn.Module = torch.nn.Tanh(),
        max_num_neighbors: int = 1000,
        aggr: str = "add",
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
        for _ in range(num_interactions):
            filter_network = MLP(
                layer_widths=[rbf_layer.num_rbf, num_filters],
                activation_func=activation,
            )

            cfconv = CFConv(
                filter_network,
                cutoff=cutoff,
                num_filters=num_filters,
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                aggr=aggr,
            )
            block = InteractionBlock(cfconv, hidden_channels, activation)
            interaction_blocks.append(block)
        output_layer_widths = (
            [hidden_channels] + output_hidden_layer_widths + [1]
        )
        output_network = MLP(output_layer_widths, activation_func=activation)
        super(StandardSchNet, self).__init__(
            embedding_layer,
            interaction_blocks,
            rbf_layer,
            output_network,
            max_num_neighbors=max_num_neighbors,
        )
