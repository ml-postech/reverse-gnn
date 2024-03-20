# Code reference:
# This script is based on graph-neural-pde by melifluous in twitter-research
# Source: https://github.com/twitter-research/graph-neural-pde.git

import torch
from torch import nn
import torch_sparse
from torch_geometric.nn.conv import MessagePassing

from utils import MaxNFEException

# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
class LaplacianODEFunc(MessagePassing):
    # currently requires in_features = out_features
    def __init__(self, args, data, device):
        super(LaplacianODEFunc, self).__init__()
        self.args = args
        self.device = device
        self.edge_index = None
        self.edge_weight = None
        self.attention_weights = None
        self.alpha_train = nn.Parameter(torch.tensor(0.0))
        self.x0 = None
        self.nfe = 0

    def sparse_multiply(self, x):
        mean_attention = self.attention_weights.mean(dim=1)
        ax = torch_sparse.spmm(
            self.edge_index, mean_attention, x.shape[0], x.shape[0], x
        )
        return ax

    def forward(self, t, x):  # the t param is needed by the ODE solver.
        if self.nfe > self.args.max_nfe:
            raise MaxNFEException
        self.nfe += 1
        ax = self.sparse_multiply(x)
        if not self.args.no_alpha_sigmoid:
            alpha = torch.sigmoid(self.alpha_train)
        else:
            alpha = self.alpha_train

        f = alpha * (ax - x)
        return f
    
    def __repr__(self):
        return self.__class__.__name__