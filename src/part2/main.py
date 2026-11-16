import argparse
import json
import statistics
from pathlib import Path

import torch
import yaml

from src.part2.data import describe, load_splits
from src.part2.train import train_run

METRICS = ("accuracy", "precision", "recall", "f1")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/part2/gnn.yaml")
    parser.add_argument("--ta", action="store_true")
    return parser.parse_args()


def summarize(runs):
    summary = {}
    for metric in METRICS:
        values = [run["test"][metric] for run in runs]
        summary[metric] = {"mean": statistics.mean(values), "std": statistics.stdev(values)}
    summary["params"] = runs[0]["params"]
    summary["best_epochs"] = [run["best_epoch"] for run in runs]
    return summary


def build_jobs(cfg):
    jobs = [("part_a", name, name, cfg["architecture"]["feature"]) for name in cfg["architecture"]["models"]]
    jobs += [("part_b", feature, cfg["ablation"]["model"], feature) for feature in cfg["ablation"]["features"]]
    return jobs


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suffix = "_ta" if args.ta else ""
    checkpoint_dir = Path(cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"device {device}", flush=True)

    splits_cache, runs_cache, curves = {}, {}, {}
    summary = {"part_a": {}, "part_b": {}}
    for section, label, model_name, feature in build_jobs(cfg):
        if feature not in splits_cache:
            splits_cache[feature] = load_splits(cfg["data_root"], cfg["dataset"], feature, cfg["standardize"])
            print(f"{feature} {describe(splits_cache[feature])}", flush=True)
        key = (model_name, feature)
        if key not in runs_cache:
            runs = []
            for seed in cfg["seeds"]:
                run = train_run(model_name, feature, seed, splits_cache[feature], cfg, device)
                name = f"{model_name}-{feature}-seed{seed}"
                torch.save(run.pop("state"), checkpoint_dir / f"{name}.pt")
                curves[name] = run
                runs.append(run)
                print(f"{name} best_epoch={run['best_epoch']} val_f1={run['best_val_f1']:.4f} test={run['test']}", flush=True)
            runs_cache[key] = runs
        summary[section][label] = summarize(runs_cache[key])

    means = json.loads(Path(cfg["template"]).read_text())
    for section, block in means.items():
        for label, entry in block.items():
            if label not in summary[section]:
                continue
            for metric in entry:
                entry[metric] = summary[section][label][metric]["mean"]
    write_json(Path(cfg["output_dir"]) / f"results{suffix}.json", means)
    write_json(Path(cfg["results_dir"]) / f"summary{suffix}.json", {"config": cfg, "summary": summary})
    write_json(Path(cfg["results_dir"]) / f"curves{suffix}.json", {"config": cfg, "runs": curves})
    print(json.dumps(means, indent=2), flush=True)


if __name__ == "__main__":
    main()
