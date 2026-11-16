import argparse
import json
import os
from pathlib import Path

import wandb


def load_env(path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--curves", default="results/part2/curves.json")
    parser.add_argument("--project", default="anlp-a3-part2")
    parser.add_argument("--entity", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    load_env(Path(".env"))
    if os.environ.get("WANDB_KEY"):
        wandb.login(key=os.environ["WANDB_KEY"])
    payload = json.loads(Path(args.curves).read_text())
    for name, run in payload["runs"].items():
        config = dict(payload["config"])
        config.update({"model": run["model"], "feature": run["feature"], "seed": run["seed"], "params": run["params"]})
        experiment = wandb.init(
            project=args.project,
            entity=args.entity,
            name=name,
            group=f"{run['model']}-{run['feature']}",
            job_type="architecture" if run["feature"] == payload["config"]["architecture"]["feature"] else "ablation",
            config=config,
        )
        for row in run["curve"]:
            experiment.log(row, step=row["epoch"])
        experiment.summary.update({f"test/{k}": v for k, v in run["test"].items()})
        experiment.summary.update({"best_epoch": run["best_epoch"], "best_val_f1": run["best_val_f1"]})
        experiment.finish()


if __name__ == "__main__":
    main()
