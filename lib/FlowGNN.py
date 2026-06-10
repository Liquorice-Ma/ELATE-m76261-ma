import math

import torch
import torch.nn as nn
import torch_scatter
import torch_sparse

from .utils import weight_initialization


class FlowGNN(nn.Module):
    """Transform the demands into compact feature vectors known as embeddings.

    FlowGNN alternates between
    - GNN layers aimed at capturing capacity constraints;
    - DNN layers aimed at capturing demand constraints.

    In LEO mode, operates on elephant flows only with augmented bi-adjacency
    matrix including identity blocks (Eq. 7 in ELATE paper).
    """

    def __init__(self, teal_env, num_layer):
        """Initialize flowGNN with the network topology.

        Args:
            teal_env: teal environment
            num_layer: num of layers in flowGNN
        """

        super(FlowGNN, self).__init__()

        self.env = teal_env
        self.num_layer = num_layer
        self.leo_mode = teal_env.leo_mode

        self.num_path = self.env.num_path
        self.device = getattr(self.env, 'device', torch.device('cpu'))

        if not self.leo_mode:
            self.edge_index = self.env.edge_index
            self.edge_index_values = self.env.edge_index_values
            self.num_path_node = self.env.num_path_node
            self.num_edge_node = self.env.num_edge_node

        self.gnn_list = []
        self.dnn_list = []
        for i in range(self.num_layer):
            self.gnn_list.append(nn.Linear(i+1, i+1))
            self.dnn_list.append(
                nn.Linear(self.num_path*(i+1), self.num_path*(i+1)))
        self.gnn_list = nn.ModuleList(self.gnn_list)
        self.dnn_list = nn.ModuleList(self.dnn_list)

        self.apply(weight_initialization)

    def forward(self, h_0):
        """Return embeddings after forward propagation.

        Args:
            h_0: initial embeddings
        """
        if self.leo_mode:
            return self._forward_leo(h_0)
        else:
            return self._forward_teal(h_0)

    def _forward_teal(self, h_0):
        """Original TEAL forward pass."""
        h_i = h_0
        for i in range(self.num_layer):

            h_i = self.gnn_list[i](h_i)
            h_i = torch_sparse.spmm(
                self.edge_index, self.edge_index_values,
                h_0.shape[0], h_0.shape[0], h_i)

            h_i_path_node = self.dnn_list[i](
                h_i[-self.num_path_node:, :].reshape(
                    self.num_path_node//self.num_path,
                    self.num_path*(i+1)))\
                .reshape(self.num_path_node, i+1)
            h_i = torch.concat(
                [h_i[:-self.num_path_node, :], h_i_path_node], axis=0)

            h_i = torch.cat([h_i, h_0], axis=-1)

        return h_i[-self.num_path_node:, :]

    def _forward_leo(self, h_0):
        """ELATE LEO forward pass with dynamic bipartite graph.

        Uses augmented bi-adjacency Ā_t (Eq. 7) which is rebuilt
        per observation since elephant flows change each time step.
        """
        edge_index = self.env.edge_index
        edge_index_values = self.env.edge_index_values
        num_path_node = self.env.num_path_node
        num_edge_node = self.env.num_edge_node

        if num_path_node == 0:
            return torch.zeros(0, self.num_layer + 1).to(self.device)

        h_i = h_0
        total_nodes = h_0.shape[0]

        for i in range(self.num_layer):
            # GNN: message passing via augmented bi-adjacency (Eq. 7)
            h_i = self.gnn_list[i](h_i)
            h_i = torch_sparse.spmm(
                edge_index, edge_index_values,
                total_nodes, total_nodes, h_i)

            # DNN: enforce flow conservation per elephant flow (Eq. 8)
            h_i_path_node = self.dnn_list[i](
                h_i[-num_path_node:, :].reshape(
                    num_path_node // self.num_path,
                    self.num_path * (i + 1)))\
                .reshape(num_path_node, i + 1)
            h_i = torch.concat(
                [h_i[:-num_path_node, :], h_i_path_node], axis=0)

            # Skip connection
            h_i = torch.cat([h_i, h_0], axis=-1)

        return h_i[-num_path_node:, :]
