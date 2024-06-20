import matplotlib.pyplot as plt 
import networkx as nx 
import pandas as pd 
import numpy as np 
import torch 
import torch_geometric
import torchmetrics 
import sknetwork as skn 
import sklearn 
import torch.nn as nn 
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree 
import torch.nn.functional as F

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add') 
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        h = self.lin(x)

        row, col = edge_index
        deg = degree(col, h.size(0), dtype=h.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(edge_index, x=h, norm=norm)
        out += self.bias 
        
        return out 

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class Net(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, dp_rate_linear=0.5) -> None:
        super().__init__()
        
        self.dp_rate = dp_rate_linear
        self.conv1 = GCNConv(c_in, c_hidden)
        self.conv2 = GCNConv(c_hidden, c_out)
        self.lin = Linear(c_hidden, c_out)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def encoder(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.relu(x)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dp_rate, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dp_rate, training=self.training)
        x = self.lin(x)
        return F.log_softmax(x, dim=-1)


