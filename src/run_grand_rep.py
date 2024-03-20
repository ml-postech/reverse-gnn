# Code references:
# 1. The structure of the code is adapted from BernNet by ivam-he
#    Source: https://github.com/ivam-he/BernNet.git

# 2. Most of arguments are inspired by graph-neural-pde by melifluous in twitter-research
#   Source: https://github.com/twitter-research/graph-neural-pde.git

import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.logging import log
from torch_geometric.nn import GCNConv

import numpy as np
from tqdm import tqdm

import shutil
import os
import os.path as osp
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import time
import random
import copy
from datetime import datetime

from heterodata_loader import Hetero
from model_grand_rep import GRANDRep
from utils import (
    random_planetoid_splits,
    random_planetoid_splits_w_few_labels,
)


def RunExp(args, dataset, data, Net, device):
    def train(model, optimizer, data):
        model.train()
        optimizer.zero_grad()
        out = model(data.x)

        if data.y.max() == 1:
            loss = F.binary_cross_entropy_with_logits(
                out[data.train_mask], data.y[data.train_mask].float()
            )
        else:
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

        model.fm.update(model.getNFE())
        model.resetNFE()
        loss.backward()
        optimizer.step()
        model.bm.update(model.getNFE())
        model.resetNFE()

        del out

    def test(model, data):
        model.eval()
        out = model(data.x)  # [node_num, class_num]
        accs, ce_losses, roc_aucs = [], [], []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            if data.y.max() == 1:
                pred = (out.sigmoid() > 0.5).float()
                loss = F.binary_cross_entropy_with_logits(
                    out[mask], data.y[mask].float()
                )
                roc_auc = roc_auc_score(
                    data.y[mask].float().cpu().detach().numpy(),
                    out[mask].cpu().detach().numpy(),
                ).item()
                roc_aucs.append(roc_auc)
            else:
                pred = out.argmax(dim=-1)
                loss = F.cross_entropy(out[mask], data.y[mask])

            accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
            ce_losses.append(loss.detach().cpu())

        return accs, ce_losses, roc_aucs

    model = GRANDRep(args, dataset, device)

    if args.dataset in ("Cora", "CiteSeer", "PubMed", "Computers", "Photo"):
        # randomly split dataset
        if args.dataset in ("Cora", "CiteSeer", "PubMed") and args.few_labels:
            permute_masks = random_planetoid_splits_w_few_labels
            data = permute_masks(data, args.seed, args.label_num)
        else:
            permute_masks = random_planetoid_splits
            data = permute_masks(data, dataset.num_classes, args.seed)
        model, data = model.to(device), data.to(device)
    elif args.dataset in (
        "squirrel",
        "squirrel-filtered",
        "chameleon",
        "chameleon-filtered",
        "roman-empire",
        "amazon-ratings",
        "minesweeper",
        "tolokers",
        "questions",
    ):
        if args.few_labels:
            raise NotImplementedError(
                "few-labels task is just implemented for Citation Graph."
            )
        model = model.to(device)
        data.to_device()
    else:
        raise NotImplementedError

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val_metric = test_metric = 0
    best_val_loss = float("inf")
    train_loss_history = []
    val_metric_history = []
    patience_count = 0

    save_name = (
        "model/{}_{}_{:.2f}_".format(Net, args.dataset, args.time)
        + datetime.now().strftime("%Y%m%d-%H%M%S")
        + ".pt"
    )

    time_run = []
    for epoch in range(1, args.epochs + 1):
        t_st = time.time()
        train(model, optimizer, data)
        time_epoch = time.time() - t_st  # each epoch train times
        time_run.append(time_epoch)
        (
            accs,
            [train_ce_loss, val_ce_loss, tmp_test_ce_loss],
            roc_aucs,
        ) = test(model, data)
        if len(roc_aucs) == 3:
            train_metric, val_metric, tmp_test_metric = roc_aucs
        else:
            train_metric, val_metric, tmp_test_metric = accs
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            test_metric = tmp_test_metric
            patience_count = 0
            torch.save(model.state_dict(), save_name)
        else:
            patience_count += 1

        log(
            Epoch=epoch,
            Loss=train_ce_loss,
            Train=train_metric,
            Val=val_metric,
            Test=test_metric,
            patience_count=patience_count,
        )

        train_loss_history.append(train_ce_loss)
        if args.early_stopping > 0 and patience_count >= args.early_stopping:
            print("Early stopping at epoch", epoch)
            break
        if np.isnan(train_ce_loss.item()):
            print("Loss is NaN")
            break

    return test_metric, best_val_metric, time_run


def main():
    # 10 fixed seeds for splits
    SEEDS = [
        1941488137,
        4198936517,
        983997847,
        4023022221,
        4019585660,
        2108550661,
        1648766618,
        629014539,
        3212139042,
        2424918363,
    ]

    print(args)
    print("---------------------------------------------")

    assert (
        args.attention_dim % args.heads == 0
    ), "Number of heads must be a factor of the dimension size"
    gnn_name = "GRAND-rep" if args.inv_time != 0 else "GRAND"

    os.makedirs("model", exist_ok=True)
    if args.dataset in ("Cora", "CiteSeer", "PubMed"):
        path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "Planetoid")
        dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
        data = dataset[0]
    elif args.dataset in ("Computers", "Photo"):
        path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "Amazon")
        dataset = Amazon(path, args.dataset, transform=T.NormalizeFeatures())
        data = dataset[0]
    elif args.dataset in (
        "squirrel",
        "squirrel-filtered",
        "chameleon",
        "chameleon-filtered",
        "roman-empire",
        "amazon-ratings",
        "minesweeper",
        "tolokers",
        "questions",
    ):
        dataset = Hetero(args, device)
        data = dataset.data
    else:
        raise NotImplementedError

    print(data)
    results = []
    time_results = []
    for RP in tqdm(range(args.runs)):
        args.seed = SEEDS[RP]
        test_metric, best_val_metric, time_run = RunExp(
            args, dataset, data, gnn_name, device
        )
        time_results.append(time_run)
        results.append([test_metric, best_val_metric])
        print(f"run_{str(RP+1)} \t test_metric: {test_metric:.4f}")
        if args.dataset in (
            "squirrel",
            "squirrel-filtered",
            "chameleon",
            "chameleon-filtered",
            "roman-empire",
            "amazon-ratings",
            "minesweeper",
            "tolokers",
            "questions",
        ):
            dataset.next_data_split()

    run_sum = 0
    epochsss = 0
    for i in time_results:
        run_sum += sum(i)
        epochsss += len(i)

    print("each run avg_time:", run_sum / (args.runs), "s")
    print("each epoch avg_time:", 1000 * run_sum / epochsss, "ms")

    test_metric_mean, val_metric_mean = np.mean(results, axis=0) * 100
    test_metric_std, val_metric_std = np.sqrt(np.var(results, axis=0)) * 100

    print(args)
    print(f"{gnn_name} on dataset {args.dataset}, in {args.runs} repeated experiment:")
    print(f"val metric mean = {val_metric_mean:.4f} ± {val_metric_std:.4f}")
    print(f"test metric mean = {test_metric_mean:.4f} ± {test_metric_std:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "Cora",
            "CiteSeer",
            "PubMed",
            "Computers",
            "Photo",
            "squirrel",
            "squirrel-filtered",
            "chameleon",
            "chameleon-filtered",
            "roman-empire",
            "amazon-ratings",
            "minesweeper",
            "tolokers",
            "questions",
        ],
        default="Cora",
    )

    parser.add_argument("--hidden-dim", type=int, default=128)

    parser.add_argument("--lr", type=float, default=0.01, help="learning rate.")
    parser.add_argument("--epochs", type=int, default=1000, help="max epochs.")
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=200,
        help="patience degree when performing early stopping. If set to negative, no early stopping.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay.")

    parser.add_argument(
        "--few-labels",
        action="store_true",
        default=False,
        help="whether to perform few-labels task.",
    )
    parser.add_argument(
        "--label-num",
        type=int,
        default=20,
        help="given label num when task is few-labels.",
    )

    parser.add_argument(
        "--seed", type=int, default=2108550661, help="seeds for random splits."
    )
    parser.add_argument("--runs", type=int, default=10, help="number of runs.")

    parser.add_argument(
        "--self-loop-weight", type=float, default=1.0, help="Weight of self-loops."
    )
    parser.add_argument(
        "--data-norm",
        type=str,
        choices=["rw", "gcn"],
        default="rw",
        help="rw for random walk, gcn for symmetric gcn norm",
    )
    parser.add_argument(
        "--loop-len", type=float, default=1.0, help="Weight of self-loops."
    )

    # ODE args
    parser.add_argument(
        "--time", type=float, default=1.0, help="forward time of ODE integrator."
    )
    parser.add_argument(
        "--inv-time", type=float, default=0.0, help="backward time of ODE integrator."
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["dopri5", "euler", "rk4", "midpoint"],
        default="dopri5",
        help="set the numerical solver: dopri5, euler, rk4, midpoint",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=1,
        help="fixed step size when using fixed step solvers e.g. rk4",
    )
    parser.add_argument(
        "--tol-scale", type=float, default=1.0, help="multiplier for atol and rtol"
    )
    parser.add_argument(
        "--max-nfe",
        type=int,
        default=5000,
        help="Maximum number of function evaluations in an epoch. Stiff ODEs will hang if not set.",
    )
    parser.add_argument(
        "--no_alpha_sigmoid",
        dest="no_alpha_sigmoid",
        action="store_true",
        help="apply sigmoid before multiplying by alpha",
    )

    # Attention args
    parser.add_argument(
        "--heads", type=int, default=4, help="number of attention heads"
    )
    parser.add_argument(
        "--attention-dim",
        type=int,
        default=64,
        help="the size to project x to before calculating att scores",
    )
    parser.add_argument(
        "--attention-type",
        type=str,
        choices=["scaled_dot", "cosine_sim", "pearson"],
        default="scaled_dot",
    )
    parser.add_argument(
        "--reweight_attention",
        dest="reweight_attention",
        action="store_true",
        help="multiply attention scores by edge weights before softmax",
    )
    parser.add_argument(
        "--square-plus", action="store_true", help="replace softmax with square plus"
    )
    parser.add_argument(
        "--attention-norm-idx",
        type=int,
        default=0,
        help="0 = normalise rows, 1 = normalise cols",
    )
    parser.add_argument(
        "--mix_features",
        dest="mix_features",
        action="store_true",
        help="apply a feature transformation xW to the ODE",
    )

    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    main()
