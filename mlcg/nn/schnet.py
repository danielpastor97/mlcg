from typing import Optional, List
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_cluster import radius_graph
from .basis import GaussianSmearing, ExpNormalSmearing, CosineCutoff
from ..geometry.internal_coordinates import compute_distances


class SchNet(nn.Module):
    """PyTorch Geometric implementation of SchNet
    Code adapted from https://github.com/rusty1s/pytorch_geometric/blob/d7d8e5e2edada182d820bbb1eec5f016f50db1e0/torch_geometric/nn/models/schnet.py#L38

    Parameters
    ----------
    embedding_layer: torch.nn.Module
        Initial embedding layer that transforms atoms/coarse grain bead
        types into embedded features
    interaction_blocks: list of torch.nn.Module or torch.nn.Sequential
        Sequential interaction blocks of the model, where each interaction
        block applies
    rbf_layer: torch.nn.Module
        The set of radial basis functions that expands pairwise distances between
        atoms/CG beads.
    max_num_neighbors: int (default=100)
        Maximum number of neighbors to return for a
        given node/atom when constructing the molecular graph during forward passes.
        This attribute is passed to the torch_cluster radius_graph routine keyword
        max_num_neighbors, which normally defaults to 32. Users should set this to
        higher values if they are using higher upper distance cutoffs and expect more
        than 32 neighbors per node/atom.
    """

    def __init__(
        self,
        embedding_layer: nn.Module,
        interaction_blocks: List[nn.Module],
        rbf_layer: nn.Module,
        max_num_neighbors: int = 1000,
    ):

        super(SchNet, self).__init__()

        self.embedding_layer = embedding_layer
        self.rbf_layer = rbf_layer
        self.max_num_neighbors = max_num_neighbors

        if isinstance(interaction_blocks, List):
            self.interaction_blocks = nn.Sequential(*interaction_blocks)
        elif instance(interaction_blocks, InteractionBlock):
            self.interaction_blocks = nn.Sequential(interaction_blocks)
        else:
            raise RuntimeError(
                "interaction_blocks must be a single InteractionBlock or "
                "a list of InteractionBlocks"
            )
        # warn the user if cutoffs in RBFs and CFConv are different
        if isinstance(self.rbf_layer, (GaussianSmearing, ExpNormalSmearing)):
            for block in self.interaction_blocks:
                if block.conv.cutoff != None:
                    if (
                        block.conv.cutoff.cutoff_upper
                        != self.rbf_layer.cutoff_upper
                    ):
                        warnings.warn(
                            "Convolution lower cutoff and RBF lower cutoff do not match."
                        )

    def reset_parameters(self):
        self.embedding_layer.reset_parameters()
        self.rbf_layer.reset_parameters()
        for block in self.interaction_blocks:
            interaction.reset_parameters()

    def forward(self, data):
        """Forward pass through the SchNet architecture.

        Parameters
        ----------
        data: torch_geometric.data.Data object
            Input data object containing batch atom/bead positions
            and atom/bead types.

        Returns
        -------
        x: torch.tensor,
           SchNet features, of shape (num_examples * num_atoms, hidden_channels)
        """
        x = self.embedding_layer(data.z)
        edge_index = radius_graph(
            data.pos,
            r=self.rbf_layer.cutoff_upper,
            batch=data.batch,
            max_num_neighbors=self.max_num_neighbors,
        )

        distances = compute_distances(data.pos, edge_index)
        rbf_expansion = self.rbf_layer(distances)

        for block in self.interaction_blocks:
            x = x + block(x, edge_index, distances, rbf_expansion)

        return x

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"embedding={self.embedding_layer}, "
            f"interaction_blocks={self.interaction_blocks}, "
            f"rbf_layer={self.rbf_layer}, "
        )


class InteractionBlock(nn.Module):
    """Interaction blocks for SchNet. Consists of atomwise transformations
    of embedded features that are continuously convolved with filters generated
    from radial basis function-expanded pairwise distances.

    Parameters
    ----------
    cfconv_layer: CFConv object
        Continuous filter convolution layer for convolutions of radial basis
        function-expanded distances with embedded features
    hidden_channels: int (default=128)
        Hidden dimension of embedded features
    activation: type
        Activation function applied to linear layer outputs
    """

    def __init__(
        self,
        cfconv_layer: nn.Module,
        hidden_channels: int = 128,
        activation: nn.Module = nn.Tanh,
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

    def forward(self, x, edge_index, edge_weight, edge_attr):
        """Forward pass through the interaction block.

        Parameters
        ----------
        x: torch.Tensor
            Embedded features of shape (num_examples, num_atoms, hidden_channels)
        edge_index: torch.Tensor
            Graph edge index tensor of shape (2, total_num_edges)
        edge_weight: torch.Tensor
            Graph edge weight (eg, distances), of shape (total_num_edges)
        edge_attr: torch.Tensor
            Graph edge attributes (eg, expanded distances), of shape (total_num_edges, num_rbf)

        Returns
        -------
        x: torch.Tensor
            Updated embedded features of shape (num_examples * num_atoms, hidden_channels)
        """

        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.activation(x)
        x = self.lin(x)
        return x


class CFConv(MessagePassing):
    """Forward pass through the interaction block.

    Parameters
    ----------
    filter_net: nn.Module
        Neural network for generating filters from expanded pairwise distances
    cutoff: nn.Module (default=None)
        Cutoff envelope to apply to pariwise distances
    in_channels: int (default=128)
        Hidden input dimensions
    out_channels: int (default=128)
        Hidden output dimensions
    num_filters: int (default=128)
        Number of filters
    aggr: str (default='add')
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
        if cutoff != None:
            self.cutoff = cutoff
        else:
            self.cutoff = None

        self.reset_parameters()

    def reset_parameters(self):
        """Method for resetting the weights of the linear
        layers according the the Xavier uniform strategy. Biases
        are set to 0.
        """

        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        """Forward pass through the interaction block.

        Parameters
        ----------
        x: torch.Tensor
            Embedded features of shape (num_examples * num_atoms, hidden_channels)
        edge_index: torch.Tensor
            Graph edge index tensor of shape (2, total_num_edges)
        edge_weight: torch.Tensor
            Graph edge weight (eg, distances), of shape (total_num_edges)
        edge_attr: torch.Tensor
            Graph edge attributes (eg, expanded distances), of shape (total_num_edges, num_rbf)

        Returns
        -------
        x: torch.Tensor
            Updated embedded features of shape (num_examples * num_atoms, hidden_channels)
        """
        if self.cutoff:
            C = self.cutoff(edge_weight)
            W = self.filter_network(edge_attr) * C.view(-1, 1)
        else:
            W = self.filter_network(edge_attr)

        x = self.lin1(x)
        # propagate_type: (x: Tensor, W: Tensor)
        # Perform the continuous filter convolution
        x = self.propagate(edge_index, x=x, W=W, size=None)
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        """Message passing operation to perform the continuous filter convolution
        through element-wise multiplcation of embedded features with the output
        of the filter network.

        Parameters
        ----------
        x_j: torch.Tensor
            Tensor of embedded features of shape (total_num_edges, hidden_channels)
        W: torch.Tensor
            Tensor of filter values of shape (total_num_edges, num_filters)
        """
        return x_j * W


def create_schnet(
    hidden_channels: int = 128,
    max_z: int = 100,
    num_filters: int = 128,
    num_interactions: int = 3,
    num_rbf: int = 50,
    rbf_type: str = "gauss",
    trainable_rbf: bool = False,
    activation: type = nn.Tanh,
    cutoff_lower: float = 0.0,
    cutoff_upper: float = 0.5,
    conv_cutoff: bool = True,
    max_num_neighbors: int = 1000,
    aggr: str = "add",
) -> SchNet:

    """Helper function to create a typical SchNet"""

    defined_bases = {"gauss": GaussianSmearing, "expnorm": ExpNormalSmearing}

    if rbf_type not in defined_bases.keys():
        raise RuntimeError(
            "Specified RBF type '{}' is not defined.".format(rbf_type)
        )
    if num_interactions < 1:
        raise RuntimeError("At least one interaction block must be specified")
    if num_rbf < 1:
        raise RuntimeError(
            "The number of RBFs must be greater than or equal to 1"
        )
    if cutoff_upper < cutoff_lower:
        raise RuntimeError(
            "Upper cutoff must be greater than or equal to lower cutoff"
        )

    embedding_layer = nn.Embedding(max_z, hidden_channels)
    rbf_layer = defined_bases[rbf_type](
        cutoff_lower, cutoff_upper, num_rbf, trainable=trainable_rbf
    )

    interaction_blocks = []
    for _ in range(num_interactions):
        filter_network = nn.Sequential(
            nn.Linear(num_rbf, num_filters),
            activation(),
            nn.Linear(num_filters, num_filters),
        )
        if conv_cutoff != None:
            conv_cutoff = CosineCutoff(
                cutoff_lower=cutoff_lower, cutoff_upper=cutoff_upper
            )
        else:
            conv_cutoff = None
        cfconv = CFConv(
            filter_network,
            num_filters=num_filters,
            cutoff=conv_cutoff,
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
        max_num_neighbors=max_num_neighbors,
    )

    return schnet
