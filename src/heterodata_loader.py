# Code reference:
# This script is based on Non-Homophily-Large-Scale by Xiuyu-Li in CUAI
# Source: https://github.com/CUAI/Non-Homophily-Large-Scale.git

import torch_geometric
from torch_geometric.utils import to_undirected

import torch
import numpy as np
import scipy.io
import os
from os import path

DATAPATH = path.abspath(path.join(path.dirname(__file__), "..")) + "/data/"


def rand_train_test_idx(label, train_prop=0.5, valid_prop=0.25, ignore_negative=True):
    """randomly splits label into train/valid/test splits"""
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num : train_num + valid_num]
    test_indices = perm[train_num + valid_num :]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx


class Data:
    def __init__(self):
        self.edge_index = None
        self.edge_attr = None
        self.x = None
        self.y = None
        self.num_features = None
        self.num_classes = None
        self.num_nodes = None
        self.train_mask = None
        self.val_mask = None
        self.test_mask = None
        self.device = None

    def get_idx_split(self, split_type="random", train_prop=0.5, valid_prop=0.25):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """

        if split_type == "random":
            # ignore_negative = False if self.name == "ogbn-proteins" else True
            ignore_negative = True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.y,
                train_prop=train_prop,
                valid_prop=valid_prop,
                ignore_negative=ignore_negative,
            )
            # split_idx = {"train": train_idx, "valid": valid_idx, "test": test_idx}
            num_nodes = self.x.shape[0]
            self.train_mask = torch.tensor(
                [True if idx in train_idx else False for idx in range(num_nodes)]
            )
            self.val_mask = torch.tensor(
                [True if idx in valid_idx else False for idx in range(num_nodes)]
            )
            self.test_mask = torch.tensor(
                [True if idx in test_idx else False for idx in range(num_nodes)]
            )
        # return train_mask, valid_mask, test_mask

    def to_device(self):
        self.edge_index = self.edge_index.to(self.device)
        self.x = self.x.to(self.device)
        self.y = self.y.to(self.device)
        self.train_mask = self.train_mask.to(self.device)
        self.val_mask = self.val_mask.to(self.device)
        self.test_mask = self.test_mask.to(self.device)
        if self.edge_attr is not None:
            self.edge_attr = self.edge_attr.to(self.device)


class Hetero:
    def __init__(self, args, device):
        print("Preparing data...")
        data = np.load(os.path.join(DATAPATH, f'{args.dataset.replace("-", "_")}.npz'))

        # fulldata = scipy.io.loadmat(f"{DATAPATH}/{name}.mat")
        self.data = Data()
        self.data.edge_index = to_undirected(
            torch.tensor(data["edges"], dtype=torch.int64).T
        )
        self.data.x = torch.from_numpy(data["node_features"])
        self.data.y = torch.from_numpy(
            np.array(data["node_labels"], dtype=int).flatten()
        )
        self.train_masks = torch.tensor(data["train_masks"])
        self.val_masks = torch.tensor(data["val_masks"])
        self.test_masks = torch.tensor(data["test_masks"])

        self.cur_data_split = 0

        self.data.train_mask = self.train_masks[self.cur_data_split]
        self.data.val_mask = self.val_masks[self.cur_data_split]
        self.data.test_mask = self.test_masks[self.cur_data_split]
        self.num_data_splits = len(self.train_masks)

        self.num_features = self.data.x.shape[-1]
        self.num_classes = self.data.y.max() + 1

        self.data.num_features = self.num_features
        self.data.num_classes = self.num_classes
        self.data.num_nodes = self.data.x.shape[0]

        self.name = args.dataset
        self.device = device
        self.data.device = device

    def next_data_split(self):
        self.cur_data_split = (self.cur_data_split + 1) % self.num_data_splits
        self.data.train_mask = self.train_masks[self.cur_data_split]
        self.data.val_mask = self.val_masks[self.cur_data_split]
        self.data.test_mask = self.test_masks[self.cur_data_split]