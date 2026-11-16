# ANLP A3

## Part 1

```bash
uv sync
bash scripts/part1/test_training.sh
bash scripts/part1/test_assn.sh
bash scripts/part1/test_assn.sh --heldout
uv run python -m src.part1.main push --model all
bash ./scripts/part1/log_wandb.sh
```

### Kaggle GPU

```bash
export KAGGLE_USERNAME=<your-kaggle-username>
bash ./scripts/part1/kaggle_push.sh src.part1.main train --model all
bash ./scripts/part1/kaggle_pull.sh
```

## Part 2

```bash
uv sync
bash ./scripts/part2/test_gnn.sh
bash ./scripts/part2/test_gnn.sh --ta
bash ./scripts/part2/log_wandb.sh
bash ./scripts/part2/log_wandb.sh --mode best
uv run python -m src.part2.push_hf --repo <hf-username>/anlp-a3-part2
```

### Kaggle GPU

```bash
export KAGGLE_USERNAME=<your-kaggle-username>
bash ./scripts/part2/kaggle_push.sh
bash ./scripts/part2/kaggle_pull.sh
```
