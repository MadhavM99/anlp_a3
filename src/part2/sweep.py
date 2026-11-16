import argparse
import itertools
import json
import statistics
from pathlib import Path

import torch
import yaml

from src.part2.data import describe, load_splits
from src.part2.main import build_jobs, write_json
from src.part2.train import train_run


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/part2/sweep.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    sweep = yaml.safe_load(Path(args.config).read_text())
    base = yaml.safe_load(Path(sweep["base"]).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    keys = sorted({(model, feature) for _, _, model, feature in build_jobs(base)})
    splits_cache = {}
    for _, feature in keys:
        if feature not in splits_cache:
            splits_cache[feature] = load_splits(base["data_root"], base["dataset"], feature, base["standardize"])
            print(f"{feature} {describe(splits_cache[feature])}", flush=True)

    names = list(sweep["grid"])
    trials = []
    for values in itertools.product(*(sweep["grid"][name] for name in names)):
        cfg = dict(base, **dict(zip(names, values)))
        scores = {}
        for model_name, feature in keys:
            vals = [
                train_run(model_name, feature, seed, splits_cache[feature], cfg, device, evaluate_test=False)["best_val_f1"]
                for seed in sweep["seeds"]
            ]
            scores[f"{model_name}-{feature}"] = {"mean": statistics.mean(vals), "min": min(vals), "values": vals}
        trial = {"params": dict(zip(names, values)), "mean_val_f1": statistics.mean(s["mean"] for s in scores.values()), "scores": scores}
        trials.append(trial)
        print(json.dumps({"params": trial["params"], "mean_val_f1": trial["mean_val_f1"], "per_config": {k: round(v["mean"], 4) for k, v in scores.items()}}), flush=True)

    best = max(trials, key=lambda trial: trial["mean_val_f1"])
    write_json(Path(sweep["output"]), {"selection": "mean best-epoch val F1 over all configurations and seeds", "best": best["params"], "trials": trials})
    print(f"best {best['params']} mean_val_f1={best['mean_val_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
