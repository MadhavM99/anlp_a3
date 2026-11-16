# Implementation Choices

## Part 2

### Data
- GossipCop is loaded with `torch_geometric.datasets.UPFD`, using its built-in 20/10/70 train/val/test split. The data is never re-split.
- Node 0 of every graph is the source news post. In a batch, the roots are found with `ptr[:-1]`.
- Bi-GCN takes the stored edge direction as top-down and its reverse as bottom-up. The run log prints a direction check for each feature type.

### Architectures
| Key | Reference | What the code does |
|---|---|---|
| `mlp` | text-only baseline | 2-layer MLP on the root's features only. No edges or retweeter nodes are used. |
| `gcn` | Kipf & Welling, 2017 | 2 symmetric-normalised GCN layers with self-loops, ReLU, and dropout before each layer. |
| `sage` | Hamilton et al., 2017 | 2 GraphSAGE-mean layers: `W·[h_v ‖ mean(h_N(v))]`, then ReLU, then L2 normalisation, in the paper's order. |
| `gat` | Veličković et al., 2018 | Layer 1 has 8 heads (concatenated), layer 2 has 1 head. ELU activation, dropout 0.6 on inputs and on attention coefficients, LeakyReLU slope 0.2, self-attention included. |
| `bigcn` | Bian et al., AAAI 2020, "Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks" | Top-down GCN on the original edges and bottom-up GCN on the reversed edges. Each branch adds the root's features back in at both layers, `[H1 ‖ X_root]` and `[H2 ‖ H1_root]`. DropEdge (0.2) is sampled separately for each direction. Each branch is mean-pooled, the two are concatenated, and a linear layer classifies. |

- The Bi-GCN link in the assignment PDF points to the wrong paper. The implementation follows the rumour-propagation Bi-GCN above, since that model is built for propagation graphs.
- The GCN, SAGE and GAT papers describe node classification. Turning node outputs into one prediction per graph follows the UPFD paper (Dou et al., SIGIR 2021): mean-pool the node embeddings, concatenate with a projection of the root's features, then pass through an MLP. The root-only MLP is the same model without the graph branch, so the gap between them measures what the propagation graph adds.
- GCN, SAGE and GAT treat the graph as undirected. Bi-GCN keeps the edge directions because the two directions are its core idea.
- GraphSAGE uses every neighbour instead of the paper's fixed-size sampling. Sampling only speeds up training on large graphs, and these graphs are small.

### Feature standardisation
- Every input feature dimension is standardised with the mean and standard deviation of the training split's nodes. The same scaling is applied to val and test. The official GraphSAGE implementation does the same (`StandardScaler` fitted on training nodes).
- The raw features are small but uneven in scale. bert and spacy have per-dimension standard deviations around 0.04–0.11; profile and content lie in [0, 1].
- Without standardisation, GraphSAGE (with its ReLU and L2 normalisation) trained badly. On bert, 4 of 5 seeds stopped at the root-only MLP's validation F1, i.e. the graph branch was unused. On content, 4 of 5 seeds collapsed to predicting a single class (val F1 0.34). With standardisation, all 40 runs train properly.
- The same preprocessing is applied to every model and every feature type, so the ablation still changes only the feature type.

### Training and evaluation
- All models share one training setup: Adam, lr 0.005, weight decay 0.01, batch size 128, 100 epochs, hidden size 128, dropout 0.5, cross-entropy loss.
- Settings specific to one architecture follow its reference paper: GAT heads and dropout, Bi-GCN DropEdge.
- The test set is evaluated with the weights from the epoch with the best validation F1.
- Precision, recall and F1 are macro-averaged, as in the UPFD paper. Set `average: binary` in `config/part2/gnn.yaml` for positive-class scores instead.
- Seeds are 0–4. The ± values are the sample standard deviation over the 5 seeds.
- The feature ablation changes only the input features. The `sage` + `bert` runs are shared between the architecture comparison and the ablation.

### Outputs
- `output/part2/results.json`: `results/part2/results_template.json` (trailing comma fixed so it parses as JSON) filled with mean test scores (`results_ta.json` with `--ta`).
- Model checkpoints are uploaded to HuggingFace with `src/part2/push_hf.py`.
- `results/part2/summary.json`: mean, standard deviation, parameter count and best epochs for each configuration.
- `results/part2/curves.json`: per-epoch train and val loss, accuracy, precision, recall and F1 for every run. `src/part2/log_wandb.py` uploads these to WandB, one run per seed, grouped by configuration.
- Compute runs in a Kaggle GPU kernel (`scripts/part2/kaggle_push.sh`), which downloads the dataset there. `scripts/part2/kaggle_pull.sh` copies the outputs back.
