import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv

from spectral_norm_gnn import spectral_norm_gnn
from utils import cal_GSL, minesweeper_plot

NORMALIZATION = {
    "None": nn.Identity,
    "LayerNorm": nn.LayerNorm,
    "BatchNorm": nn.BatchNorm1d,
}


class resgnn_block(nn.Module):
    def __init__(self, args, in_channels):
        super(resgnn_block, self).__init__()
        if args.net.endswith("resgcn"):
            self.body = GCNConv(in_channels, in_channels)
        elif args.net.endswith("resgat"):
            self.body = GATConv(in_channels, in_channels, args.heads, concat=False)

    def forward(self, x, edge_index):
        Fx = self.body(x, edge_index).relu()
        return x + Fx


class resgnn(nn.Module):
    def __init__(self, args, dataset, device=torch.device("cpu")):
        super(resgnn, self).__init__()

        self.args = args
        self.depth = args.depth
        self.num_classes = dataset.num_classes
        self.num_features = dataset.data.num_features
        self.num_nodes = dataset.data.num_nodes
        self.device = device
        self.hidden_dim = args.hidden_dim

        self.in_mlp = not args.no_in_mlp
        self.out_mlp = not args.no_out_mlp

        if self.in_mlp:
            self.in_ = nn.Linear(self.num_features, self.hidden_dim)
        elif args.net.endswith("resgcn"):
            self.in_ = GCNConv(self.num_features, self.hidden_dim)
        elif args.net.endswith("resgat"):
            self.in_ = GATConv(
                self.num_features, self.hidden_dim // args.heads, args.heads
            )

        self.dropout = nn.Dropout(p=args.dropout)

        self.stack = nn.ModuleList()
        self.indp_block = args.indp_block
        for _ in range(self.indp_block):
            self.stack.append(resgnn_block(args, self.hidden_dim))

        self.out_norm = NORMALIZATION[args.otptnorm](self.hidden_dim)
        out_dim = 1 if self.num_classes == 2 else self.num_classes
        if self.out_mlp:
            out = []
            out.append(nn.Linear(args.hidden_dim, args.hidden_dim))
            out.append(nn.ReLU())
            out.append(nn.Linear(args.hidden_dim, out_dim))
            self.out = nn.Sequential(*out)
        elif args.net.endswith("resgcn"):
            self.out = GCNConv(self.hidden_dim, out_dim)
        elif args.net.endswith("resgat"):
            self.out = GATConv(self.hidden_dim, out_dim, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = self.in_(x) if self.in_mlp else self.in_(x, edge_index)
        x = self.dropout(x)
        x = x.relu()

        for depth_idx in range(self.depth):
            x = self.stack[depth_idx % self.indp_block](x, edge_index)

        x = self.out_norm(x)
        if self.out_mlp:
            x = self.out(x)
        else:
            x = self.out(x, edge_index)
        return x.squeeze(1)

    def get_gsl(self, x, edge_index):
        with torch.no_grad():
            x = self.in_(x) if self.in_mlp else self.in_(x, edge_index)
            x = self.dropout(x)
            x = x.relu()

            GSL_by_depth = []
            depth = []
            for depth_idx in range(self.depth):
                x = self.stack[depth_idx % self.indp_block](x, edge_index)
                if (depth_idx & depth_idx + 1) == 0:
                    GSL_by_depth.append(cal_GSL(x))
                    depth.append(depth_idx + 1)

        return GSL_by_depth, depth, None, None

    def map_to_prob(self, x):
        with torch.no_grad():
            x = self.out_norm(x)
            x = self.out(x)

        return torch.sigmoid(x).squeeze(1)

    def mine_plot(
        self,
        x,
        edge_index,
        file_name,
        sample_idx=None,
        sample_size=None,
    ):
        with torch.no_grad():
            x = self.in_(x) if self.in_mlp else self.in_(x, edge_index)
            x = self.dropout(x)
            x = x.relu()
            minesweeper_plot(
                self.map_to_prob(x),
                file_name + "after_encoder",
                sample_idx=sample_idx,
                sample_size=sample_size,
            )

            for depth_idx in range(self.depth):
                x = self.stack[depth_idx % self.indp_block](x, edge_index)
                if (depth_idx & depth_idx + 1) == 0:
                    minesweeper_plot(
                        self.map_to_prob(x),
                        file_name + "afterf{}".format(depth_idx + 1),
                        sample_idx=sample_idx,
                        sample_size=sample_size,
                    )


class iresgnn_block(nn.Module):
    def __init__(self, args, in_channels, dataset):
        super(iresgnn_block, self).__init__()
        if args.net.endswith("resgcn"):
            self.body = spectral_norm_gnn(
                GCNConv(in_channels, in_channels),
                net=args.net,
                coeff=args.coeff,
                name="weight",
            )
        elif args.net.endswith("resgat"):
            self.body = spectral_norm_gnn(
                GATConv(in_channels, in_channels, args.heads, concat=False),
                net=args.net,
                coeff=args.coeff,
                name="weight",
                heads=args.heads,
            )
        self.maxiter = args.maxiter
        self.converge_threshold = args.converge_threshold

    def forward(self, x, edge_index):
        Fx = self.body(x, edge_index).relu()
        return x + Fx

    def inverse(self, x, edge_index):
        x_input = x
        x_history = [x]
        for iter_idx in range(self.maxiter):
            summand = self.body(x, edge_index).relu()
            x = x_input - summand
            x_history.append(x)
            diff = (x_history[-1] - x_history[-2]).abs().mean()
            if diff < self.converge_threshold:
                break

        return x


class iresgnn(nn.Module):
    def __init__(self, args, dataset, device=torch.device("cpu")):
        super(iresgnn, self).__init__()

        self.args = args
        self.depth = args.depth
        self.inv_depth = args.inv_depth
        self.num_classes = dataset.num_classes
        self.num_features = dataset.data.num_features
        self.num_nodes = dataset.data.num_nodes
        self.device = device
        self.hidden_dim = args.hidden_dim

        self.in_mlp = not args.no_in_mlp
        self.out_mlp = not args.no_out_mlp

        if self.in_mlp:
            self.in_ = nn.Linear(self.num_features, self.hidden_dim)
        elif args.net.endswith("resgcn"):
            self.in_ = GCNConv(self.num_features, self.hidden_dim)
        elif args.net.endswith("resgat"):
            self.in_ = GATConv(
                self.num_features, self.hidden_dim // args.heads, args.heads
            )

        self.dropout = nn.Dropout(p=args.dropout)

        self.stack = nn.ModuleList()
        self.indp_block = args.indp_block
        for _ in range(self.indp_block):
            self.stack.append(iresgnn_block(args, self.hidden_dim, dataset))

        self.out_norm = NORMALIZATION[args.otptnorm](self.hidden_dim * 2)
        out_dim = 1 if self.num_classes == 2 else self.num_classes
        if args.plot_ms:
            self.out = nn.Linear(self.hidden_dim * 2, out_dim)
        elif self.out_mlp:
            out = []
            out.append(nn.Linear(args.hidden_dim * 2, args.hidden_dim))
            out.append(nn.ReLU())
            out.append(nn.Linear(args.hidden_dim, out_dim))
            self.out = nn.Sequential(*out)
        elif args.net.endswith("resgcn"):
            self.out = GCNConv(self.hidden_dim * 2, out_dim)
        elif args.net.endswith("resgat"):
            self.out = GATConv(self.hidden_dim * 2, out_dim, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = self.in_(x) if self.in_mlp else self.in_(x, edge_index)
        x = self.dropout(x)
        x = x.relu()

        z_f = x
        for depth_idx in range(self.depth):
            # print("depth_idx: {}, block_idx: {}".format(depth_idx, depth_idx%self.indp_block))
            z_f = self.stack[depth_idx % self.indp_block](z_f, edge_index)
        z_b = x
        for inv_depth_idx in range(self.inv_depth):
            # print("depth_idx: {}, block_idx: {}".format(inv_depth_idx, self.indp_block-1-inv_depth_idx%self.indp_block))
            z_b = self.stack[
                self.indp_block - 1 - inv_depth_idx % self.indp_block
            ].inverse(z_b, edge_index)

        x = torch.cat([z_f, z_b], dim=1)

        x = self.out_norm(x)
        if self.args.plot_ms or self.out_mlp:
            x = self.out(x)
        else:
            x = self.out(x, edge_index)
        return x.squeeze(1)

    def get_gsl(self, x, edge_index):
        with torch.no_grad():
            x = self.in_(x) if self.in_mlp else self.in_(x, edge_index)
            x = self.dropout(x)
            x = x.relu()

            z_f = x
            GSL_by_depth = []
            depth = []
            for depth_idx in range(self.depth):
                z_f = self.stack[depth_idx % self.indp_block](z_f, edge_index)
                if (depth_idx & depth_idx + 1) == 0:
                    GSL_by_depth.append(cal_GSL(z_f))
                    depth.append(depth_idx + 1)

            z_b = x
            GSL_by_inv_depth = []
            inv_depth = []
            for inv_depth_idx in range(self.inv_depth):
                z_b = self.stack[
                    self.indp_block - 1 - inv_depth_idx % self.indp_block
                ].inverse(z_b, edge_index)
                if (inv_depth_idx & inv_depth_idx + 1) == 0:
                    GSL_by_inv_depth.append(cal_GSL(z_b))
                    inv_depth.append(inv_depth_idx + 1)

        return GSL_by_depth, depth, GSL_by_inv_depth, inv_depth
    
    def split_out(self, x, edge_index, forward):
        with torch.no_grad():
            x = self.in_(x) if self.in_mlp else self.in_(x, edge_index)
            x = self.dropout(x)
            x = x.relu()

            if forward:
                z_f = x
                for depth_idx in range(self.depth):
                    # print("depth_idx: {}, block_idx: {}".format(depth_idx, depth_idx%self.indp_block))
                    z_f = self.stack[depth_idx % self.indp_block](z_f, edge_index)
                x = torch.cat([z_f, torch.zeros_like(z_f).to(device=z_f.device)], 1)
            else:
                z_b = x
                for inv_depth_idx in range(self.inv_depth):
                    # print("depth_idx: {}, block_idx: {}".format(inv_depth_idx, self.indp_block-1-inv_depth_idx%self.indp_block))
                    z_b = self.stack[
                        self.indp_block - 1 - inv_depth_idx % self.indp_block
                    ].inverse(z_b, edge_index)
                x = torch.cat([torch.zeros_like(z_b).to(device=z_b.device), z_b], dim=1)

            x = self.out_norm(x)
            x = self.out(x)
            return x.squeeze(1)

    def map_to_prob(self, x):
        with torch.no_grad():
            x = self.out_norm(x)
            x = self.out(x)

            return torch.sigmoid(x).squeeze(1)

    def mine_plot(
        self,
        x,
        edge_index,
        file_name,
        sample_idx=None,
        sample_size=None,
    ):
        with torch.no_grad():
            x = self.in_(x) if self.in_mlp else self.in_(x, edge_index)
            x = self.dropout(x)
            x = x.relu()
            minesweeper_plot(
                self.map_to_prob(torch.cat([x, x], 1)),
                file_name + "after_encoder",
                sample_idx=sample_idx,
                sample_size=sample_size,
            )

            z_f = x
            for depth_idx in range(self.depth):
                # print("depth_idx: {}, block_idx: {}".format(depth_idx, depth_idx%self.indp_block))
                z_f = self.stack[depth_idx % self.indp_block](z_f, edge_index)
                if (depth_idx & depth_idx + 1) == 0:
                    minesweeper_plot(
                        self.map_to_prob(
                            torch.cat(
                                [z_f, torch.zeros_like(z_f).to(device=z_f.device)], 1
                            )
                        ),
                        file_name + "afterf{}".format(depth_idx + 1),
                        sample_idx=sample_idx,
                        sample_size=sample_size,
                    )
            z_b = x
            for inv_depth_idx in range(self.inv_depth):
                # print("depth_idx: {}, block_idx: {}".format(inv_depth_idx, self.indp_block-1-inv_depth_idx%self.indp_block))
                z_b = self.stack[
                    self.indp_block - 1 - inv_depth_idx % self.indp_block
                ].inverse(z_b, edge_index)
                if (inv_depth_idx & inv_depth_idx + 1) == 0:
                    minesweeper_plot(
                        self.map_to_prob(
                            torch.cat(
                                [torch.zeros_like(z_b).to(device=z_b.device), z_b],
                                dim=1,
                            )
                        ),
                        file_name + "afterb{}".format(inv_depth_idx + 1),
                        sample_idx=sample_idx,
                        sample_size=sample_size,
                    )

            x = torch.cat([z_f, z_b], dim=1)
            minesweeper_plot(
                self.map_to_prob(x),
                file_name + "final_pred",
                sample_idx=sample_idx,
                sample_size=sample_size,
            )

    def check_inverse(self, x, edge_index):
        #####################################################################################
        #### By examining the difference below,
        #### you can assess whether the inverse function has been implemented correctly.
        #### If the inverse function is well-implemented,
        #### the difference value should be sufficiently small.
        #####################################################################################
        with torch.no_grad():
            x = self.in_(x) if self.in_mlp else self.in_(x, edge_index)

            z_f = x
            diff = []
            for depth_idx in range(self.depth):
                prev_z = z_f
                z_f = self.stack[depth_idx % self.indp_block](z_f, edge_index)
                diff.append(
                    (
                        self.stack[depth_idx % self.indp_block].inverse(z_f, edge_index)
                        - prev_z
                    )
                    .abs()
                    .mean()
                )
            z_b = x
            for inv_depth_idx in range(self.inv_depth):
                prev_z = z_b
                z_b = self.stack[
                    self.indp_block - 1 - inv_depth_idx % self.indp_block
                ].inverse(z_b, edge_index)
                diff.append(
                    (
                        self.stack[
                            self.indp_block - 1 - inv_depth_idx % self.indp_block
                        ](z_b, edge_index)
                        - prev_z
                    )
                    .abs()
                    .mean()
                )
            diff_mean = torch.stack(diff).mean()
            print("diff_mean: {}".format(diff_mean))
            return diff_mean

    def check_inverse2(self, x, edge_index):
        #####################################################################################
        #### By examining the difference below,
        #### you can assess whether the inverse function has been implemented correctly.
        #### If the inverse function is well-implemented,
        #### the difference value should be sufficiently small.
        #####################################################################################
        with torch.no_grad():
            x = self.in_(x) if self.in_mlp else self.in_(x, edge_index)
            
            z_f = x
            forward_history = [z_f]
            for depth_idx in range(self.depth):
                z_f = self.stack[depth_idx % self.indp_block](z_f, edge_index)
                forward_history.append(z_f)
            z_b = z_f
            backward_history = [z_b]
            for inv_depth_idx in range(self.depth):
                z_b = self.stack[
                    self.indp_block - 1 - inv_depth_idx % self.indp_block
                ].inverse(z_b, edge_index)
                backward_history.append(z_b)
            backward_history.reverse()
            diff = [
                (ori - res).abs().mean()
                for ori, res in zip(forward_history, backward_history)
            ]
            print("diff: {}".format(diff))
            return diff
