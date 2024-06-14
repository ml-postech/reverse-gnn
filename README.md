# Mitigating Oversmoothing Through Reverse Process of GNNs for Heterophilic Graphs

This repository contains a PyTorch implementation of ["Mitigating Oversmoothing Through Reverse Process of GNNs for Heterophilic Graphs"](https://arxiv.org/abs/2403.10543) (ICML, 2024).

### Python environment setup with Conda

```bash
conda create --name rep python=3.8
conda activate rep
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install pyg -c pyg
pip install seaborn
pip install wandb
conda install pytorch-sparse -c pyg
pip install torchdiffeq
pip install networkx
```
### Running the code

You can run the code using script files in `script/` directory.  
For example, you can run GRAND-rep for the squirrel dataset with `./script/grand_rep.sh`.  
Also, you can run GCN-rep for the squirrel dataset with `./script/gcn_rep.sh`.

### Plot the graph

If you want to plot the graph smoothing level, use the `--plot-gsl` argument.  
If you want to visualize a result for the minesweeper dataset, use the `--plot-ms` argument.  
You can directly run `./script/plot.sh`.
