# Code reference:
# This script includes functions adapted from graph-neural-pde by melifluous in twitter-research
# Source: https://github.com/twitter-research/graph-neural-pde.git

import torch
from torch import Tensor
from torch_geometric.utils import add_remaining_self_loops, scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add, scatter, segment_csr, gather_csr

from typing import Optional
import numpy as np

import matplotlib.pyplot as plt

class MaxNFEException(Exception):
    pass


def get_rw_adj(
    edge_index, edge_weight=None, norm_dim=1, fill_value=0.0, num_nodes=None, dtype=None
):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones(
            (edge_index.size(1),), dtype=dtype, device=edge_index.device
        )

    if not fill_value == 0:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes
        )
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    indices = row if norm_dim == 0 else col
    deg = scatter_add(edge_weight, indices, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-1)
    edge_weight = (
        deg_inv_sqrt[indices] * edge_weight
        if norm_dim == 0
        else edge_weight * deg_inv_sqrt[indices]
    )
    return edge_index, edge_weight


from typing import Optional
import torch
from torch import Tensor
from torch_scatter import scatter, segment_csr, gather_csr


# https://twitter.com/jon_barron/status/1387167648669048833?s=12
# @torch.jit.script
def squareplus(
    src: Tensor,
    index: Optional[Tensor],
    ptr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Tensor:
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        ptr (LongTensor, optional): If given, computes the softmax based on
            sorted inputs in CSR representation. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    out = src - src.max()
    # out = out.exp()
    out = (out + torch.sqrt(out**2 + 4)) / 2

    if ptr is not None:
        out_sum = gather_csr(segment_csr(out, ptr, reduce="sum"), ptr)
    elif index is not None:
        N = maybe_num_nodes(index, num_nodes)
        out_sum = scatter(out, index, dim=0, dim_size=N, reduce="sum")[index]
    else:
        raise NotImplementedError

    return out / (out_sum + 1e-16)


# Counter of forward and backward passes.
class Meter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.sum = 0
        self.cnt = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.cnt += 1

    def get_average(self):
        if self.cnt == 0:
            return 0
        return self.sum / self.cnt

    def get_value(self):
        return self.val


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask


def random_planetoid_splits_w_few_labels(data, seed, num_per_class=20):
    ########################################
    # In this code, we use "num_per_class" labeled data per class for training, 500 node data for validation, and 1,000 node data for test.
    # In GRAND++, they use 1,500 node data for train and validation, and rest of the data for test.
    ########################################
    rnd_state = np.random.RandomState(seed)
    num_nodes = data.y.shape[0]
    test_idx = list(rnd_state.choice(num_nodes, 1000, replace=False))
    development_idx = np.array([i for i in np.arange(num_nodes) if i not in test_idx])

    train_idx = []
    for c in range(data.y.max() + 1):
        class_idx = development_idx[np.where(data.y[development_idx].cpu() == c)[0]]
        train_idx.extend(rnd_state.choice(class_idx, num_per_class, replace=False))

    val_idx = [i for i in development_idx if i not in train_idx][:500]

    data.train_mask = index_to_mask(train_idx, data.num_nodes)
    data.val_mask = index_to_mask(val_idx, data.num_nodes)
    data.test_mask = index_to_mask(test_idx, data.num_nodes)

    if (
        data.train_mask.float() + data.val_mask.float() + data.test_mask.float()
    ).max().item() > 1:
        raise ValueError("Overlap between train, val and test")

    return data


def random_planetoid_splits(data, num_classes, seed=12134):
    train_lb = int(round(0.6 * len(data.y)))
    val_lb = int(round(0.2 * len(data.y)))

    index = [i for i in range(0, data.y.shape[0])]
    rnd_state = np.random.RandomState(seed)
    train_idx = rnd_state.choice(index, train_lb, replace=False)
    rest_index = [i for i in index if i not in train_idx]
    val_idx = rnd_state.choice(rest_index, val_lb, replace=False)
    test_idx = [i for i in rest_index if i not in val_idx]
    # print(test_idx)

    data.train_mask = index_to_mask(train_idx, size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx, size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx, size=data.num_nodes)

    return data


def cal_degree_hat(edge_index, edge_weight, num_nodes, dtype):
    if edge_weight is not None:
        edge_weight = edge_weight.squeeze()
    edge_index, edge_weight = add_remaining_self_loops(
        edge_index, edge_weight, 1.0, num_nodes
    )
    if edge_weight is None:
        edge_weight = torch.ones(
            (edge_index.size(1),), dtype=dtype, device=edge_index.device
        )

    row, col = edge_index[0], edge_index[1]
    # degree = torch_geometric.utils.degree(row)
    degree = scatter(edge_weight, col, dim=0, dim_size=num_nodes, reduce="sum")
    degree_hat = degree ** (-0.5)
    degree_hat.masked_fill_(degree_hat == float("inf"), 0)

    degree_hat_idx = torch.tensor(list(range(num_nodes))).unsqueeze(0).repeat(2, 1)
    degree_hat_tensor = torch.sparse_coo_tensor(
        degree_hat_idx.to(device=degree_hat.device),
        degree_hat,
        (num_nodes, num_nodes),
    )

    return degree_hat_tensor


def cal_A_hat(edge_index, edge_weight, num_nodes):
    edge_index, edge_weight = add_remaining_self_loops(
        edge_index, edge_weight, 1.0, num_nodes
    )
    adj_hat_tensor = torch.sparse_coo_tensor(
        edge_index, torch.ones(edge_index.shape[-1]).to(device=edge_index.device)
    )

    return adj_hat_tensor


def cal_GSL(x):
    with torch.no_grad():
        num_nodes = x.shape[0]
        cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        batch_size = 1000 if num_nodes > 1000 else num_nodes
        for i in range(0, num_nodes, batch_size):
            now_node_SL = cos(
                x.unsqueeze(0), x[i : i + batch_size].unsqueeze(1)
            ).fill_diagonal_(0).sum(1) / float(num_nodes - 1.0)
            if i == 0:
                node_SL = now_node_SL
            else:
                node_SL = torch.cat(
                    [
                        node_SL,
                        now_node_SL,
                    ],
                )
        GSL = node_SL.sum() / num_nodes

    return GSL.item()


def plot_GSL(gsl_list, depth_list, gsl_list2=None, depth_list2=None, graph_name=None):
    plt.figure(figsize=(6, 4))
    plt.plot(depth_list, gsl_list, marker="o")
    if gsl_list2 is not None:
        plt.plot(depth_list2, gsl_list2, marker="*")

    graph_name = "GSL" if graph_name is None else graph_name
    plt.title(graph_name)
    plt.xlabel("Depth")
    plt.ylabel("Graph Smoohing Level")
    plt.grid(True)
    plt.ylim(0, 1)

    plt.savefig("plot/{}.png".format(graph_name))
    plt.close()

def minesweeper_plot(X, file_name, sample_idx=None, sample_size=None):
    fig, ax = plt.subplots(figsize=(4, 4))
    Z = X.reshape(100, 100)
    if sample_idx is not None:
        assert (
            sample_size is not None
        ), "sample_size should be given when sample_idx is given"
        idx_i, idx_j = sample_idx // (101 - sample_size), sample_idx % (
            101 - sample_size
        )
        Z = Z[idx_i : idx_i + sample_size, idx_j : idx_j + sample_size]  # sampled image
        save_name = "plot/sampled[{},{}]_{}.png".format(idx_i, idx_j, file_name)
        file_name = "sampled[{},{}]_{}".format(idx_i, idx_j, file_name)

        cax = ax.imshow(
            Z.cpu().detach().numpy(),
            cmap="binary",
            interpolation="nearest",
            origin="upper",
            extent=[idx_j, idx_j + sample_size, idx_i + sample_size, idx_i],
            vmin=0,  
            vmax=1,  
        )
    else:
        save_name = "plot/entire_{}.png".format(file_name)

        cax = ax.imshow(
            Z.cpu().detach().numpy(),
            cmap="binary",
            interpolation="nearest",
            origin="upper",
            vmin=0,  
            vmax=1,  
        )

    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.savefig(save_name)
    ax.cla()
    plt.close()
