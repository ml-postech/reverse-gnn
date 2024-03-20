# Code reference:
# This script is based on graph-neural-pde by melifluous in twitter-research
# Source: https://github.com/twitter-research/graph-neural-pde.git

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing

from ode_function import LaplacianODEFunc
from ode_block import AttODEblock
from utils import Meter


# Define the GNN model.
class GRANDRep(MessagePassing):
    def __init__(self, args, dataset, device=torch.device("cpu")):
        super(GRANDRep, self).__init__()

        self.args = args
        self.delta_t = args.step_size
        self.num_classes = dataset.num_classes
        self.num_features = dataset.data.num_features
        self.num_nodes = dataset.data.num_nodes
        self.device = device
        self.fm = Meter()
        self.bm = Meter()
        self.hidden_dim = args.hidden_dim

        self.T = args.time
        self.T_inv = args.inv_time

        time_tensor = torch.tensor([0, self.T]).to(device)
        inv_time_tensor = (
            torch.tensor([0, -self.T_inv]).to(device) if self.T_inv != 0 else None
        )
        self.f = LaplacianODEFunc
        self.odeblock = AttODEblock(
            self.f,
            args,
            dataset.data,
            device,
            t=time_tensor,
            t_inv=inv_time_tensor,
        ).to(device)

        self.m1 = nn.Linear(self.num_features, args.hidden_dim)

        m2_dim = args.hidden_dim if inv_time_tensor is None else args.hidden_dim * 2
        self.m21 = nn.Linear(m2_dim, args.hidden_dim)
        if self.num_classes == 2:
            self.m22 = nn.Linear(args.hidden_dim, 1)
        else:
            self.m22 = nn.Linear(args.hidden_dim, self.num_classes)

    def forward(self, x):
        x = self.m1(x).relu()

        self.odeblock.set_x0(x)

        z = self.odeblock(x).relu()

        # Decode each node embedding to get node label.
        z = self.m21(z).relu()
        z = self.m22(z)

        return z.squeeze(1)

    def getNFE(self):
        return self.odeblock.odefunc.nfe

    def resetNFE(self):
        self.odeblock.odefunc.nfe = 0

    def reset(self):
        self.m1.reset_parameters()
        self.m21.reset_parameters()
        self.m22.reset_parameters()

    def __repr__(self):
        return self.__class__.__name__
