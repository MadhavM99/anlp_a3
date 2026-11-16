import argparse
import json
import os
from pathlib import Path

import wandb

METRICS = ("loss", "accuracy", "precision", "recall", "f1")
SPLITS = ("train", "val")
TEST_METRICS = ("accuracy", "precision", "recall", "f1")
OVERLAYS = ("val/f1", "val/loss", "train/f1", "train/loss")


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
    parser.add_argument("--project", default="anlp_a3_part2")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--mode", choices=("runs", "sections", "best"), default="runs")
    return parser.parse_args()


def ensure_project(project, entity):
    api = wandb.Api()
    entity = entity or api.default_entity
    if not any(existing.name == project for existing in api.projects(entity)):
        api.create_project(project, entity)
    return entity


def replay_run(name, run, base_config, project, entity, group, job_type, tags):
    config = dict(base_config)
    config.update({"model": run["model"], "feature": run["feature"], "seed": run["seed"], "params": run["params"]})
    experiment = wandb.init(
        project=project, entity=entity, name=name, group=group, job_type=job_type, tags=tags, config=config
    )
    experiment.define_metric("epoch")
    experiment.define_metric("*", step_metric="epoch")
    experiment.define_metric("val/f1", summary="max")
    experiment.define_metric("val/loss", summary="min")
    best_f1, best_epoch = -1.0, 0
    for row in run["curve"]:
        if row["val/f1"] > best_f1:
            best_f1, best_epoch = row["val/f1"], row["epoch"]
        experiment.log(dict(row, lr=base_config["lr"], **{"best/val_f1": best_f1, "best/epoch": best_epoch}))
    experiment.log({"epoch": run["curve"][-1]["epoch"], **{f"test/{k}": v for k, v in run["test"].items()}})
    final = run["curve"][-1]
    experiment.summary.update({f"final/{split}_{metric}": final[f"{split}/{metric}"] for split in SPLITS for metric in METRICS})
    experiment.summary.update({"best_epoch": run["best_epoch"], "best_val_f1": run["best_val_f1"]})
    experiment.finish()


def build_sections(config):
    architecture = [(model, model, config["architecture"]["feature"]) for model in config["architecture"]["models"]]
    ablation = [(feature, config["ablation"]["model"], feature) for feature in config["ablation"]["features"]]
    return {"architecture_comparison": architecture, "feature_ablation": ablation}


def best_seed(runs, model, feature):
    candidates = [run for run in runs.values() if run["model"] == model and run["feature"] == feature]
    return max(candidates, key=lambda run: (run["best_val_f1"], -run["seed"]))


def log_section(section, entries, payload, project, entity):
    selected = {label: best_seed(payload["runs"], model, feature) for label, model, feature in entries}
    config = dict(payload["config"], section=section, selected_seeds={label: run["seed"] for label, run in selected.items()})
    experiment = wandb.init(
        project=project, entity=entity, name=f"{section}-best-seed", group=section, job_type=section, tags=[section, "best-seed"], config=config
    )
    experiment.define_metric("epoch")
    experiment.define_metric("*", step_metric="epoch")
    num_epochs = min(len(run["curve"]) for run in selected.values())
    epochs = list(range(1, num_epochs + 1))
    for index in range(num_epochs):
        row = {"epoch": index + 1}
        for label, run in selected.items():
            point = run["curve"][index]
            for split in SPLITS:
                for metric in METRICS:
                    row[f"{split}_{metric}/{label}"] = point[f"{split}/{metric}"]
        experiment.log(row)

    columns = ["config", "seed", "best_epoch", "best_val_f1"]
    columns += [f"final_{split}_{metric}" for split in SPLITS for metric in METRICS]
    columns += [f"test_{metric}" for metric in TEST_METRICS]
    rows = []
    for label, run in selected.items():
        final = run["curve"][num_epochs - 1]
        rows.append(
            [label, run["seed"], run["best_epoch"], run["best_val_f1"]]
            + [final[f"{split}/{metric}"] for split in SPLITS for metric in METRICS]
            + [run["test"][metric] for metric in TEST_METRICS]
        )
        experiment.summary.update({f"final/{label}/{split}_{metric}": final[f"{split}/{metric}"] for split in SPLITS for metric in METRICS})
        experiment.summary.update({f"test/{label}/{metric}": run["test"][metric] for metric in TEST_METRICS})

    charts = {"final_metrics": wandb.Table(columns=columns, data=rows)}
    for key in OVERLAYS:
        charts[f"compare/{key.replace('/', '_')}"] = wandb.plot.line_series(
            xs=epochs,
            ys=[[point[key] for point in run["curve"][:num_epochs]] for run in selected.values()],
            keys=list(selected),
            title=f"{section}: {key} (best seed per config)",
            xname="epoch",
        )
    experiment.log(dict(charts, epoch=num_epochs))
    experiment.finish()


def main():
    args = parse_args()
    load_env(Path(".env"))
    if os.environ.get("WANDB_KEY"):
        wandb.login(key=os.environ["WANDB_KEY"])
    entity = ensure_project(args.project, args.entity)
    payload = json.loads(Path(args.curves).read_text())
    if args.mode == "sections":
        for section, entries in build_sections(payload["config"]).items():
            log_section(section, entries, payload, args.project, entity)
        return
    if args.mode == "best":
        for section, entries in build_sections(payload["config"]).items():
            for label, model, feature in entries:
                run = best_seed(payload["runs"], model, feature)
                name = f"{section}/{label}-seed{run['seed']}"
                tags = [section, "best-seed", run["model"], run["feature"]]
                replay_run(name, run, payload["config"], args.project, entity, f"{section}-best-seed", section, tags)
        return
    architecture_feature = payload["config"]["architecture"]["feature"]
    for name, run in payload["runs"].items():
        job_type = "architecture" if run["feature"] == architecture_feature else "ablation"
        group = f"{run['model']}-{run['feature']}"
        replay_run(name, run, payload["config"], args.project, entity, group, job_type, [run["model"], run["feature"]])


if __name__ == "__main__":
    main()
