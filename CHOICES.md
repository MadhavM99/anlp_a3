# Implementation Choices

## Part 1

### Formulation
- The inserted punctuation `!'+,-./:;<=>?[]_{}` is removed from the ciphertext. In every document of every split, the stripped ciphertext is then exactly as long as the plaintext (`results/part1/probe.json`). This turns decryption into a per-position task with no generation step: every cipher position gets a 27-way prediction over space and a–z.
- The cipher depends on the plaintext seen so far, not just the current character (`results/part1/probe_dependency.json`). Position 0 is a pure substitution. Further into a document, knowing the cipher character alone predicts the plaintext character less and less often, and knowing the previous plaintext characters helps. Earlier ciphertext adds nothing on its own. All models are therefore causal stacks that can build this running state.
- Training uses 64-character patches with 4 characters of left context, so each input is 68 tokens and each patch has 64 labels. At a document's start, the left context is padding, which tells the model it is at position 0. At inference, 64-character windows slide over the document and their predictions are joined. Every document in the test and held-out sets gets an output line.
- Training reads the first 100k training documents and uses at most 1,000 validation documents for model selection. JSONL files are read by byte offsets. The vocabularies are built from the training slice: 63 cipher symbols plus padding and unknown, and 27 plaintext symbols.

### Models (all written from scratch in PyTorch)
- All four models share the same shell: embedding, 6 blocks with pre-norm (RMSNorm) and residual connections, a final RMSNorm, and a linear head. d_model is 256 and dropout 0.1. Each has about 4.77M parameters, within 8% of one another.
- **Transformer:** causal multi-head self-attention (4 heads) with rotary position embeddings, plus a GELU MLP with d_ff 1024.
- **Mamba (Gu & Dao, 2023):** in_proj splits into x and z; then a causal depthwise conv (width 4) with SiLU; the input-dependent Δ, B and C projections (Δ rank ⌈d/16⌉); S4D-real initialisation of A (A_n = −n); and the Δ bias initialised log-uniform in [0.001, 0.1]. Then the selective scan h_t = exp(Δ_t A) h_{t−1} + Δ_t B_t x_t, y_t = C_t h_t + D x_t, gated by SiLU(z), and out_proj. As in the paper, the Mamba block has no separate MLP. Its inner width is set to 960 (expansion 3.75), so each block has the same parameter count as a Transformer layer. d_state is 8. The scan runs in fp32 as a recurrence over time steps; there is no Triton kernel.
- **DeltaNet (Yang et al., 2024):** q, k, v projections, each with a short causal convolution (width 4) and SiLU. q and k are L2-normalised. β = sigmoid(W_β x) per head. The delta rule S_t = S_{t−1} + β_t k_t (v_t − S_{t−1}ᵀ k_t)ᵀ is computed with the paper's chunkwise parallel form (chunk 32): the WY representation is obtained by solving the triangular system (I + tril(diag(β) K Kᵀ, −1)) W = diag(β) K in fp32, and the state is carried across chunks. The output then goes through a per-head RMSNorm and the output projection, followed by a GELU MLP with d_ff 1024. `src/part1/smoke.py` checks the chunkwise form against the plain recurrence.
- **Hybrid:** Mamba and DeltaNet alternate, starting with Mamba, as committed in `docs/DESIGN.md` before implementation began.

### Training and evaluation
- Seed 0. AdamW (β = 0.9, 0.95). Learning rate 8e-4 for the Transformer and DeltaNet, 3e-4 for Mamba and the hybrid (no grid search). 500 warmup steps then cosine decay, weight decay 0.01 (not applied to norms, embeddings, A_log or D), gradient clip 1.0, label smoothing 0.1, batch size 16, 10k steps. Mixed precision is fp16 with a gradient scaler; the scans and triangular solves stay in fp32.
- Validation loss is measured every 500 steps on at most 60 batches, and the checkpoint with the lowest validation loss is kept. Decoding metrics are computed only once, on the selected checkpoint, after training. That end-of-training evaluation is capped at 100 documents, 16 chunks per document and 8 minutes per split. `scripts/part1/test_assn.sh` and the held-out run decode every document in full.
- Metrics:
  - **Character accuracy:** the share of positions predicted correctly.
  - **Word accuracy:** a word counts only if every character in it is right; words come from splitting the reference on spaces.
  - **Precision, recall and F1:** macro-averaged over the 27 plaintext characters.
  - **Levenshtein distance:** both the mean per document and the mean normalised by document length.
  - **FLOPs:** counted with PyTorch's FLOP counter for a forward pass over one 68-token window, plus an analytic count for the scan and triangular-solve operations it doesn't track.

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
