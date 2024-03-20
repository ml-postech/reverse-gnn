# Code references:
# The structure of the code is adapted from BernNet by ivam-he
# Source: https://github.com/ivam-he/BernNet.git

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.logging import log

import numpy as np
from tqdm import tqdm

import os
import os.path as osp
import argparse
from sklearn.metrics import roc_auc_score

import time
from datetime import datetime

from heterodata_loader import Hetero
from model_resgnn_rep import iresgnn, resgnn

from utils import (
    random_planetoid_splits,
    random_planetoid_splits_w_few_labels,
    plot_GSL,
    minesweeper_plot,
)


def RunExp(args, dataset, data, Net, device, save_name=None):
    def train(model, optimizer, data):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)

        if data.y.max() == 1:
            loss = F.binary_cross_entropy_with_logits(
                out[data.train_mask], data.y[data.train_mask].float()
            )
        else:
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

        loss.backward()
        optimizer.step()

        del out

    def test(model, data):
        model.eval()
        out = model(data.x, data.edge_index)  # [node_num, class_num]
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

    model = Net(args, dataset, device)

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
        "model/{}_{}_{}_{}_".format(args.net, args.dataset, args.depth, args.inv_depth)
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
            torch.save(model, save_name)
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
            print("The sum of epochs:", epoch)
            break
        if np.isnan(train_ce_loss.item()):
            print("The sum of epochs:", epoch)
            break

    bestmodel = Net(args, dataset, device)
    bestmodel = torch.load(save_name)
    if args.plot_gsl:
        os.makedirs("plot", exist_ok=True)
        GSL_list1, depth1, GSL_list2, depth2 = bestmodel.get_gsl(
            data.x, data.edge_index
        )
        if GSL_list2 is not None:
            plot_GSL(
                GSL_list1,
                depth1,
                GSL_list2,
                depth2,
                graph_name="{}_{}_depth{}_invdepth{}".format(
                    args.net, args.dataset, depth1[-1], depth2[-1]
                ),
            )
        else:
            plot_GSL(
                GSL_list1,
                depth1,
                graph_name="{}_{}_depth{}".format(args.net, args.dataset, depth1[-1]),
            )
    if args.plot_ms != "None":
        assert (
            args.dataset == "minesweeper"
        ), "Only minesweeper dataset can be plotted with plot-ms argument."
        os.makedirs("plot", exist_ok=True)
        if args.net.startswith("i"):
            bestmodel.eval()
            out_type = ["all", "forward", "backward"]
            out_list = []
            out_list.append(bestmodel(data.x, data.edge_index))
            out_list.append(bestmodel.split_out(data.x, data.edge_index, forward=True))
            out_list.append(bestmodel.split_out(data.x, data.edge_index, forward=False))
            acc_per_sample_list = []
            for out_idx in range(len(out_list)):
                out = out_list[out_idx]
                pred = (out.sigmoid() > 0.5).float()
                acc_per_cell = (pred == data.y).float().reshape(100, 100)

                sample_size = args.plot_ms_size
                kernel_size = sample_size
                kernel = torch.ones((1, 1, kernel_size, kernel_size)) / (
                    kernel_size**2
                )
                acc_per_cell_4d = acc_per_cell.unsqueeze(0).unsqueeze(0)
                acc_per_sample = F.conv2d(acc_per_cell_4d, kernel.to(device), stride=1)[
                    0, 0
                ]
                acc_per_sample_list.append(acc_per_sample)

            sn = args.plot_sam_num
            for out_idx in range(len(out_list)):
                if args.plot_ms == "rand":
                    plot_indices = torch.from_numpy(
                        np.random.choice(
                            len(acc_per_sample_list[out_idx].flatten()), sn
                        )
                    )
                elif args.plot_ms == "best":
                    plot_indices = torch.topk(
                        acc_per_sample_list[out_idx].flatten(), sn
                    ).indices.cpu()
                elif args.plot_ms == "worst":
                    plot_indices = torch.topk(
                        acc_per_sample_list[out_idx].flatten(), sn, largest=False
                    ).indices.cpu()
                both_values = torch.tensor(
                    [
                        acc_per_sample_list[0].flatten()[plot_indices[i]]
                        for i in range(sn)
                    ]
                )
                forward_values = torch.tensor(
                    [
                        acc_per_sample_list[1].flatten()[plot_indices[i]]
                        for i in range(sn)
                    ]
                )
                backward_values = torch.tensor(
                    [
                        acc_per_sample_list[2].flatten()[plot_indices[i]]
                        for i in range(sn)
                    ]
                )
                W = (torch.tensor([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).to(device)) / 5.0
                for idx in range(sn):
                    minesweeper_plot(
                        data.y,
                        file_name="label",
                        sample_idx=plot_indices[idx],
                        sample_size=sample_size,
                    )
                    minesweeper_plot(
                        1.0 - data.x @ W,
                        file_name="rep",
                        sample_idx=plot_indices[idx],
                        sample_size=sample_size,
                    )
                    bestmodel.mine_plot(
                        data.x,
                        data.edge_index,
                        file_name="{}_in_{}_tacc{:.2f}_facc{:.2f}_bacc{:.2f}_".format(
                            out_type[out_idx],
                            args.plot_ms,
                            both_values[idx],
                            forward_values[idx],
                            backward_values[idx],
                        ),
                        sample_idx=plot_indices[idx],
                        sample_size=sample_size,
                    )
        else:
            bestmodel.eval()
            out = bestmodel(data.x, data.edge_index)
            pred = (out.sigmoid() > 0.5).float()
            acc_per_cell = (pred == data.y).float().reshape(100, 100)

            sample_size = args.plot_ms_size
            kernel_size = sample_size
            kernel = torch.ones((1, 1, kernel_size, kernel_size)) / (kernel_size**2)
            acc_per_cell_4d = acc_per_cell.unsqueeze(0).unsqueeze(0)
            acc_per_sample = F.conv2d(acc_per_cell_4d, kernel.to(device), stride=1)[
                0, 0
            ]

            sn = args.plot_sam_num
            if args.plot_ms == "rand":
                plot_indices = torch.from_numpy(
                    np.random.choice(len(acc_per_sample.flatten()), sn)
                )
                plot_values = torch.tensor(
                    [acc_per_sample.flatten()[plot_indices[i]] for i in range(sn)]
                )
            elif args.plot_ms == "best":
                plot_values, plot_indices = torch.topk(acc_per_sample.flatten(), sn)
                plot_values, plot_indices = plot_values.cpu(), plot_indices.cpu()
            elif args.plot_ms == "worst":
                plot_values, plot_indices = torch.topk(
                    acc_per_sample.flatten(), sn, largest=False
                )
                plot_values, plot_indices = plot_values.cpu(), plot_indices.cpu()

            W = (torch.tensor([2.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).to(device)) / 5.0
            for idx in range(sn):
                minesweeper_plot(
                    data.y,
                    file_name="label",
                    sample_idx=plot_indices[idx],
                    sample_size=sample_size,
                )
                minesweeper_plot(
                    1.0 - data.x @ W,
                    file_name="rep",
                    sample_idx=plot_indices[idx],
                    sample_size=sample_size,
                )
                bestmodel.mine_plot(
                    data.x,
                    data.edge_index,
                    file_name="{}_acc{:2f}_".format(args.plot_ms, plot_values[idx]),
                    sample_idx=plot_indices[idx],
                    sample_size=sample_size,
                )

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

    gnn_name = args.net
    if args.no_in_mlp and args.net.endswith("gat"):
        assert (
            args.hidden_dim % args.heads == 0
        ), "hidden_dim must be divisible by heads when using gat as encoder"
    assert args.depth % args.indp_block == 0, "depth must be divisible by indp_block"
    if gnn_name.startswith("ires"):
        assert (
            args.inv_depth % args.indp_block == 0
        ), "inv_depth must be divisible by indp_block"
        Net = iresgnn
    elif gnn_name.startswith("res"):
        Net = resgnn
        args.inv_depth = 0
    else:
        raise NotImplementedError

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
            args, dataset, data, Net, device
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
        default="minesweeper",
    )

    parser.add_argument(
        "--net",
        type=str,
        choices=["iresgcn", "resgcn", "iresgat", "resgat"],
        default="iresgcn",
    )
    parser.add_argument(
        "--depth", type=int, default=16, help="depth of the forward gnn."
    )
    parser.add_argument(
        "--inv-depth", type=int, default=256, help="depth of the reverse gnn."
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument(
        "--no-in-mlp",
        action="store_true",
        default=False,
        help="whether to use mlp as encoder.",
    )
    parser.add_argument(
        "--no-out-mlp",
        action="store_true",
        default=False,
        help="whether to use mlp as decoder.",
    )
    parser.add_argument(
        "--heads", type=int, default=1, help="number of attention heads"
    )
    parser.add_argument(
        "--indp-block",
        type=int,
        default=1,
        help="number of independent blocks, if using weight sharing, set to 1.",
    )

    parser.add_argument("--lr", type=float, default=0.005, help="learning rate.")
    parser.add_argument("--epochs", type=int, default=1000, help="max epochs.")
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=100,
        help="patience degree when performing early stopping. If set to negative, no early stopping.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--otptnorm",
        type=str,
        choices=["None", "LayerNorm", "BatchNorm"],
        default="None",
    )

    # fixed-point-iteration args
    parser.add_argument(
        "--coeff",
        type=float,
        default=0.999,
        help="scaling coefficient for fixed-point iteration.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=10,
        help="max iteration for fixed-point iteration.",
    )
    parser.add_argument(
        "--converge-threshold",
        type=float,
        default=1e-5,
        help="convergence threshold for fixed-point iteration.",
    )

    parser.add_argument(
        "--few-labels",
        action="store_true",
        default=False,
        help="wheter to perform few-labels task.",
    )
    parser.add_argument(
        "--label-num",
        type=int,
        default=20,
        help="given label number per class when performing few-labels task.",
    )

    parser.add_argument(
        "--seed", type=int, default=2108550661, help="seeds for random splits."
    )
    parser.add_argument("--runs", type=int, default=1, help="number of runs.")

    parser.add_argument(
        "--plot-gsl",
        action="store_true",
        default=False,
        help="whether to plot Graph Smoothing Level.",
    )
    parser.add_argument(
        "--plot-ms",
        type=str,
        choices=["None", "rand", "best", "worst"],
        default="None",
        help="whether to plot Minesweeper. if set to None, no plotting. if set to rand, plot random samples. if set to best, plot samples with best accuracy. if set to worst, plot samples with worst accuracy.",
    )
    parser.add_argument(
        "--plot-ms-size",
        type=int,
        default=10,
        help="size of the minesweeper plot. if set to n, plot n x n sample.",
    )
    parser.add_argument(
        "--plot-sam-num", type=int, default=1, help="number of samples to plot."
    )

    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    main()
