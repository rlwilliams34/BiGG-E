# BiGG-E: Extensions to the BiGG Graph Generation Framework

This repository contains code for several models to weighted graph generation, including:

- **BiGG-E**: Joint modeling of topology and edge weights 
- **BiGG-GCN**: A GCN-based two-stage edge-weight model  
- **BiGG-MLP**: A baseline using MLP-based weight prediction  
- **Adjacency-LSTM (Adj-LSTM)**: An autoregressive adjacency matrix LSTM model  
---

## Project Structure

```
BiGG-E-Repo/
├── bigg/     # Clone from https://github.com/google-research/google-research/tree/master/bigg
├── extensions/
│   ├── ADJ_LSTM/          # Adjacency-LSTM and ER baselines
│   ├── BiGG_E/            # BiGG-E and other BiGG-based extensions
│   ├── bigg-results/      # Pretrained model checkpoints
│   ├── train_graphs/      # Preprocessed training graphs
│   └── evaluate/          # Evaluation code for generated/test graphs
```

---

## Installation

Navigate to the `extensions/` folder and run:

```bash
pip install -e .
```

You will need `gcc` and the CUDA toolkit (if using GPU).  

Be sure to add the BiGG repository to your `PYTHONPATH`:  
https://github.com/google-research/google-research/tree/master/bigg

---

## Data Preparation

To generate training, validation, and test splits, go to `./synthetic_gen/scripts/` and run the corresponding file. For example, to generate tree graphs, run:

```bash
./run_tree_gen.sh
```

A script is provided for each graph type.
---

## Training and Sampling

### BiGG-E

To train models, go to `./BiGG_E/experiment/scripts`, and run the corresponding file. For example, to train BiGG-E on weighted lobster graphs, run

```bash
cd extensions/BiGG_E/experiment/scripts
./run_lobster.sh
```

To sample from the trained model:

```bash
./run_lobster.sh -phase test -epoch_load -1
```

To specify a specific epoch checkpoint, specify `epoch_load` with the corresponding epoch number. `epoch_load -1` defaults to the final epoch save.

---

### BiGG-MLP and BiGG-GCN

To run BiGG-MLP, modify the script to:

```bash
method=BiGG-MLP
```

To run BiGG-GCN, modify the script to:

```bash
model=BiGG-GCN
```

Then run the script as in the BiGG-E section above.

---

### Adjacency-LSTM

To train:

```bash
cd extensions/ADJ_LSTM/experiment/scripts
./run_lobster.sh
```

To evaluate a pre-trained model:

```bash
./run_lobster.sh -phase test -epoch_load -1
```

---

### Erdős–Rényi Baseline

To run a baseline ER model:

```bash
cd extensions/ADJ_LSTM/experiment/scripts
./run_baseline.sh GRAPH_TYPE
```

Where `GRAPH_TYPE` is one of: `tree`, `lobster`, `er`, `joint` or `db`.  
Example:

```bash
./run_baseline.sh tree
```

---

## Scalability Run

For the scalability runs, specify "method", number of leaves "num_leaves", number of epochs, and when to plateu. For example, running:

```bash
./run_scalability.sh -method BiGG-E -num_leaves 50 -num_epochs 100 -epoch_plateu 50
```

will run for 100 epochs on 50-leaf trees. Default batch size is 20. Adjust for larger graphs.

To sample from a trained model, simply load:

```bash
./run_scalability.sh -method BiGG-E -num_leaves 50 -epoch_load -1
```


## Evaluation

Evaluation code is adapted from:

- [GraphRNN](https://github.com/JiaxuanYou/graph-generation)
- [GRAN](https://github.com/lrjconan/GRAN)

These provide topological MMD metrics (degree, clustering, orbit, and spectral).  
This repo also includes MMD metrics for weighted graphs:
- Weighted degree
- Weighted Laplacian spectra
- Marginal edge weight distributions

To evaluate a trained model, run it in `-phase test` after training or sampling.

To use ORCA for orbit statistics, follow instructions from the GraphRNN repository for compilation.

---

## Code Attribution

This project builds upon existing open-source work in the graph generation community. We include and adapt components from the following:

- **BiGG** ([Google Research](https://github.com/google-research/google-research/tree/master/bigg))  
  Apache 2.0 License  
  Portions of this codebase (e.g., `FenwickTree`, `RecurTreeGen`, input preprocessing) are adapted from BiGG and clearly marked in-file.

- **GraphRNN** ([Jiaxuan You et al.](https://github.com/JiaxuanYou/graph-generation))  
  MIT License  
  Evaluation utilities (e.g., MMD code) are adapted and extended.

- **GRAN** ([Lirui Jia et al.](https://github.com/lrjconan/GRAN))  
  MIT License  
  Additional evaluation utilities are adapted.

All adaptations are credited in the corresponding Python files.  
---
