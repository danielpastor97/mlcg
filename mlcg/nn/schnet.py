from typing import Optional, List
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from ..neighbor_list.neighbor_list import atomic_data2neighbor_list
from ..data.atomic_data import AtomicData
from .radial_basis import GaussianBasis, ExpNormalBasis
from .cutoff import CosineCutoff
from ..geometry.internal_coordinates import compute_distances


class SchNet(nn.Module):
    r"""PyTorch Geometric implementation of SchNet
    Code adapted from `[PT_geom_schnet]_`  which is based on the architecture
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
    cutoff:
        Cutoff function to apply to basis-expanded distances before filter
        generation.
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

    def __init__(
        self,
        embedding_layer: nn.Module,
        interaction_blocks: List[nn.Module],
        rbf_layer: nn.Module,
        cutoff: nn.Module,
        output_network: nn.Module,
        self_interaction: bool = False,
        max_num_neighbors: int = 1000,
    ):

        super(SchNet, self).__init__()

        self.embedding_layer = embedding_layer
        self.rbf_layer = rbf_layer
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.self_interaction = self_interaction

        if isinstance(interaction_blocks, List):
            self.interaction_blocks = nn.Sequential(*interaction_blocks)
        elif isinstance(interaction_blocks, InteractionBlock):
            self.interaction_blocks = nn.Sequential(interaction_blocks)
        else:
            raise RuntimeError(
                "interaction_blocks must be a single InteractionBlock or "
                "a list of InteractionBlocks"
            )

        if rbf_layer.cutoff_lower != None and self.cutoff.cutoff_lower != None:
            if self.cutoff.cutoff_lower != self.rbf_layer.cutoff_lower:
                warnings.warn(
                    "Cutoff function lower cutoff, {}, and radial basis function "
                    " lower cutoff, {}, do not match.".format(
                        self.cutoff.cutoff_lower, self.rbf_layer.cutoff_lower
                    )
                )
        if rbf_layer.cutoff_upper != None and self.cutoff.cutoff_upper != None:
            if self.cutoff.cutoff_upper != self.rbf_layer.cutoff_upper:
                warnings.warn(
                    "Cutoff function upper cutoff, {}, and radial basis function "
                    " upper cutoff, {}, do not match.".format(
                        self.cutoff.cutoff_upper, self.rbf_layer.cutoff_upper
                    )
                )

        self.output_network = output_network

    def reset_parameters(self):
        """Method for resetting linear layers in each SchNet component"""
        for layer in self.embedding_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.fill_(0)
        self.rbf_layer.reset_parameters()
        for block in self.interaction_blocks:
            interaction.reset_parameters()
        for layer in self.output_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.fill_(0)

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
        x = self.embedding_layer(data.atomic_types)

        data.neighbor_list = atomic_data2neighbor_list(
            data, self.cutoff.cutoff_upper, self.self_interaction
        )

        distances = compute_distances(
            data.pos,
            data["neighbor_list"]["index_mapping"],
            data["neighbor_list"]["cell_shifts"],
        )
        rbf_expansion = self.rbf_layer(distances)
        if self.cutoff != None:
            rbf_expansion = rbf_expansion * self.cutoff(distances)

        for block in self.interaction_blocks:
            x = x + block(x, edge_index, distances, rbf_expansion)
        energy = self.output_network(x)
        data.energy = energy
        return data

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"embedding={self.embedding_layer}, "
            f"interaction_blocks={self.interaction_blocks}, "
            f"rbf_layer={self.rbf_layer}, "
            f"cutoff={self.cutoff},"
            f"self_interaction={self.self_interaction}, "
            f"outpu_network={self.output_network}, "
        )


class InteractionBlock(nn.Module):
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
        cfconv_layer: nn.Module,
        hidden_channels: int = 128,
        activation: type = nn.Tanh,
    ):
        super(InteractionBlock, self).__init__()
        self.conv = cfconv_layer
        self.activation = activation()
        self.lin = nn.Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

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
    r"""Forward pass through the interaction block.

    Parameters
    ----------
    filter_net:
        Neural network for generating filters from expanded pairwise distances
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
        filter_network: nn.Module,
        cutoff: Optional[nn.Module] = None,
        in_channels: int = 128,
        out_channels: int = 128,
        num_filters: int = 128,
        aggr: str = "add",
    ):
        super(CFConv, self).__init__(aggr=aggr)
        self.lin1 = nn.Linear(in_channels, num_filters, bias=False)
        self.lin2 = nn.Linear(num_filters, out_channels)
        self.filter_network = filter_network

        self.reset_parameters()

    def reset_parameters(self):
        r"""Method for resetting the weights of the linear
        layers according the the Xavier uniform strategy. Biases
        are set to 0.
        """

        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

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
        W = self.filter_network(edge_attr)

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


def create_schnet(
    rbf_layer: nn.Module,
    cutoff: nn.Module,
    output_network: nn.Module,
    hidden_channels: int = 128,
    max_z: int = 100,
    num_filters: int = 128,
    num_interactions: int = 3,
    activation: type = nn.Tanh,
    cutoff_lower: float = 0.0,
    cutoff_upper: float = 0.5,
    max_num_neighbors: int = 1000,
    aggr: str = "add",
) -> SchNet:

    r"""Helper function to create a typical SchNet"""

    if num_interactions < 1:
        raise RuntimeError("At least one interaction block must be specified")

    embedding_layer = nn.Embedding(max_z, hidden_channels)

    interaction_blocks = []
    for _ in range(num_interactions):
        filter_network = nn.Sequential(
            nn.Linear(num_rbf, num_filters),
            activation(),
            nn.Linear(num_filters, num_filters),
        )
        cfconv = CFConv(
            filter_network,
            num_filters=num_filters,
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            aggr=aggr,
        )
        block = InteractionBlock(cfconv, hidden_channels, activation)
        interaction_blocks.append(block)

    schnet = SchNet(
        embedding_layer,
        interaction_blocks,
        rbf_layer,
        output_network,
        cutoff,
        max_num_neighbors=max_num_neighbors,
    )

    return schnet
