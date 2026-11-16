# ANLP A3

## Part 2

```bash
uv sync
bash ./scripts/part2/test_gnn.sh
bash ./scripts/part2/test_gnn.sh --ta
uv run python -m src.part2.log_wandb --curves results/part2/curves.json
uv run python -m src.part2.push_hf --repo <hf-username>/anlp-a3-part2
```

### Kaggle GPU

```bash
export KAGGLE_USERNAME=<your-kaggle-username>
bash ./scripts/part2/kaggle_push.sh
bash ./scripts/part2/kaggle_pull.sh
```
