# Code reference:
# This script is based on graph-neural-pde by melifluous in twitter-research
# Source: https://github.com/twitter-research/graph-neural-pde.git

import torch
from torch import nn
from torchdiffeq import odeint
from torch_geometric.utils import softmax

import numpy as np

from utils import get_rw_adj, squareplus


class SpGraphTransAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(
        self, in_features, out_features, args, device, concat=True, edge_weights=None
    ):
        super(SpGraphTransAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.alpha = args.leaky_relu_slope
        self.concat = concat
        self.device = device
        self.args = args
        self.h = int(args.heads)
        self.edge_weights = edge_weights

        try:
            self.attention_dim = args.attention_dim
        except KeyError:
            self.attention_dim = out_features

        assert (
            self.attention_dim % self.h == 0
        ), "Number of heads ({}) must be a factor of the dimension size ({})".format(
            self.h, self.attention_dim
        )
        self.d_k = self.attention_dim // self.h

        if self.args.attention_type == "exp_kernel":
            self.output_var = nn.Parameter(torch.ones(1))
            self.lengthscale = nn.Parameter(torch.ones(1))

        self.Q = nn.Linear(in_features, self.attention_dim)
        self.init_weights(self.Q)

        self.K = nn.Linear(in_features, self.attention_dim)
        self.init_weights(self.K)

        self.activation = nn.Sigmoid()  # nn.LeakyReLU(self.alpha)

        self.Wout = nn.Linear(self.d_k, in_features)
        self.init_weights(self.Wout)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            # nn.init.xavier_uniform_(m.weight, gain=1.414)
            # m.bias.data.fill_(0.01)
            nn.init.constant_(m.weight, 1e-5)

    def forward(self, x, edge):
        """
        x might be [features, augmentation, positional encoding, labels]
        """
        q = self.Q(x)
        k = self.K(x)

        # perform linear operation and split into h heads

        k = k.view(-1, self.h, self.d_k)
        q = q.view(-1, self.h, self.d_k)

        # transpose to get dimensions [n_nodes, attention_dim, n_heads]

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)

        src = q[edge[0, :], :, :]
        dst_k = k[edge[1, :], :, :]

        if self.args.attention_type == "scaled_dot":
            prods = torch.sum(src * dst_k, dim=1) / np.sqrt(self.d_k)
        elif self.args.attention_type == "cosine_sim":
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-5)
            prods = cos(src, dst_k)
        elif self.args.attention_type == "pearson":
            src_mu = torch.mean(src, dim=1, keepdim=True)
            dst_mu = torch.mean(dst_k, dim=1, keepdim=True)
            src = src - src_mu
            dst_k = dst_k - dst_mu
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-5)
            prods = cos(src, dst_k)

        if self.args.reweight_attention and self.edge_weights is not None:
            prods = prods * self.edge_weights.unsqueeze(dim=1)
        if self.args.square_plus:
            attention = squareplus(prods, edge[self.args.attention_norm_idx])
        else:
            attention = softmax(prods, edge[self.args.attention_norm_idx])
        return attention

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class AttODEblock(nn.Module):
    def __init__(
        self,
        odefunc,
        args,
        data,
        device,
        t=torch.tensor([0, 1]),
        t_inv=None,
    ):
        super(AttODEblock, self).__init__()
        self.args = args
        self.t = t
        self.t_inv = t_inv

        self.odefunc = odefunc(args, data, device)
        edge_index, edge_weight = get_rw_adj(
            data.edge_index,
            edge_weight=data.edge_attr,
            norm_dim=1,
            fill_value=args.self_loop_weight,
            num_nodes=data.num_nodes,
            dtype=data.x.dtype,
        )
        self.odefunc.edge_index = edge_index.to(device)
        self.odefunc.edge_weight = edge_weight.to(device)

        self.train_integrator = odeint
        self.test_integrator = odeint
        self.set_tol()

        self.multihead_att_layer = SpGraphTransAttentionLayer(
            args.hidden_dim,
            args.hidden_dim,
            args,
            device,
            edge_weights=self.odefunc.edge_weight,
        ).to(device)

    def forward(self, x):
        t = self.t.type_as(x)
        inv_t = self.t_inv.type_as(x) if self.t_inv is not None else None
        self.odefunc.attention_weights = self.get_attention_weights(x)
        integrator = self.train_integrator if self.training else self.test_integrator

        func = self.odefunc
        state = x

        if inv_t is not None:
            state_dt = integrator(
                func,
                state,
                t,
                method=self.args.method,
                options={"step_size": self.args.step_size},
                atol=self.atol,
                rtol=self.rtol,
            )
            state_dt_inv = integrator(
                func,
                state,
                inv_t,
                method=self.args.method,
                options={"step_size": self.args.step_size},
                atol=self.atol,
                rtol=self.rtol,
            )
            z, z_inv = state_dt[1], state_dt_inv[1]
            return torch.cat([z, z_inv], 1)
        else:
            state_dt = integrator(
                func,
                state,
                t,
                method=self.args.method,
                options={"step_size": self.args.step_size},
                atol=self.atol,
                rtol=self.rtol,
            )

            z = state_dt[1]
            return z

    def set_x0(self, x0):
        self.odefunc.x0 = x0.clone().detach()

    def set_tol(self):
        self.atol = self.args.tol_scale * 1e-7
        self.rtol = self.args.tol_scale * 1e-9

    def reset_tol(self):
        self.atol = 1e-7
        self.rtol = 1e-9
        self.atol_adjoint = 1e-7
        self.rtol_adjoint = 1e-9

    def set_time(self, time):
        self.t = torch.tensor([0, time]).to(self.device)
        if self.t_inv is not None:
            self.t_inv = torch.tensor([0, -time]).to(self.device)

    def get_attention_weights(self, x):
        attention = self.multihead_att_layer(x, self.odefunc.edge_index)
        return attention

    def __repr__(self):
        return (
            self.__class__.__name__
            + "( Time Interval "
            + str(self.t[0].item())
            + " -> "
            + str(self.t[1].item())
            + ")"
        )
