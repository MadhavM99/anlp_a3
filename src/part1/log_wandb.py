import argparse
import json
import os
from pathlib import Path

import wandb

from src.part1.hub import load_env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/part1")
    parser.add_argument("--project", default="anlp_a3_part1")
    parser.add_argument("--entity", default=None)
    return parser.parse_args()


def ensure_project(project, entity):
    api = wandb.Api()
    entity = entity or api.default_entity
    if not any(existing.name == project for existing in api.projects(entity)):
        api.create_project(project, entity)
    return entity


def replay(curves, summary, project, entity):
    name = curves["model"]
    config = dict(curves["config"], model=name, params=curves["params"])
    experiment = wandb.init(project=project, entity=entity, name=name, group="part1", job_type="train", tags=[name], config=config)
    experiment.define_metric("step")
    experiment.define_metric("*", step_metric="step")
    experiment.define_metric("val/loss", summary="min")
    for row in curves["curve"]:
        experiment.log(row)
    final = {"step": curves["curve"][-1]["step"]}
    for split in ("validation", "test"):
        for key, value in (summary.get(split) or {}).items():
            final[f"{split}/{key}"] = value
    experiment.log(final)
    experiment.summary.update({key: summary[key] for key in ("params", "forward_flops_per_window", "best_step", "best_val_loss", "train_minutes") if key in summary})
    experiment.finish()


def main():
    args = parse_args()
    load_env(Path(".env"))
    if os.environ.get("WANDB_KEY"):
        wandb.login(key=os.environ["WANDB_KEY"])
    entity = ensure_project(args.project, args.entity)
    root = Path(args.results)
    for path in sorted(root.glob("curves_*.json")):
        curves = json.loads(path.read_text())
        summary_path = root / f"summary_{curves['model']}.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        replay(curves, summary, args.project, entity)


if __name__ == "__main__":
    main()
