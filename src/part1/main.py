import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.flop_counter import FlopCounterMode

from src.part1.data import PairCorpus, PatchDataset, build_vocabs, heldout_paths, split_paths
from src.part1.decode import decrypt_corpus
from src.part1.hub import fetch_checkpoint, load_checkpoint, push_checkpoints, save_checkpoint
from src.part1.models import build_model
from src.part1.train import set_seed, train_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train", "evaluate", "heldout", "push"))
    parser.add_argument("--config", default="config/part1/base.yaml")
    parser.add_argument("--model", default="all")
    parser.add_argument("--split", default="test")
    parser.add_argument("--source", choices=("hf", "local"), default="hf")
    parser.add_argument("--set", nargs="*", default=[])
    return parser.parse_args()


def load_config(path, overrides):
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides:
        key, value = item.split("=", 1)
        cfg[key] = yaml.safe_load(value)
    return cfg


def model_names(cfg, requested):
    return list(cfg["models"]) if requested == "all" else requested.split(",")


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def count_flops(model, cfg, device):
    x = torch.zeros(1, cfg["context"] + cfg["patch"], dtype=torch.long, device=device)
    model.eval()
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        model(x)
    counted = counter.get_total_flops()
    extra = model.extra_flops(x.size(1)) if hasattr(model, "extra_flops") else 0
    return {"forward_flops_per_window": counted + extra, "counted_matmul_flops": counted, "analytic_extra_flops": extra, "window_tokens": x.size(1)}


def model_config(cfg, name, cipher_vocab, plain_vocab):
    return dict(cfg["model"], name=name, input_vocab=len(cipher_vocab), output_vocab=len(plain_vocab))


def evaluate_split(model, cipher_vocab, plain_vocab, cfg, device, split, max_docs=None, max_chunks=None, max_minutes=None):
    cipher_path, plain_path = split_paths(cfg["data_root"], split)
    corpus = PairCorpus(cipher_path, plain_path, drop_misaligned=False)
    return decrypt_corpus(model, corpus, cipher_vocab, plain_vocab, cfg, device, max_docs, max_chunks, max_minutes)


def learning_rate(cfg, name):
    return cfg["lr"][name] if isinstance(cfg["lr"], dict) else cfg["lr"]


def run_train(cfg, names, device):
    train_corpus = PairCorpus(*split_paths(cfg["data_root"], "train"), limit=cfg["max_docs"])
    val_corpus = PairCorpus(*split_paths(cfg["data_root"], "validation"), limit=cfg["val_max_docs"])
    cipher_vocab, plain_vocab = build_vocabs(train_corpus)
    print(json.dumps({"train_docs": len(train_corpus), "train_dropped": train_corpus.dropped, "val_docs": len(val_corpus), "val_dropped": val_corpus.dropped, "cipher_vocab": cipher_vocab.to_list(), "plain_vocab": plain_vocab.to_list()}), flush=True)
    train_set = PatchDataset(train_corpus, cipher_vocab, plain_vocab, cfg["patch"], cfg["context"])
    val_set = PatchDataset(val_corpus, cipher_vocab, plain_vocab, cfg["patch"], cfg["context"])
    summary = {}
    for name in names:
        set_seed(cfg["seed"])
        model_cfg = model_config(cfg, name, cipher_vocab, plain_vocab)
        model = build_model(model_cfg).to(device)
        params = sum(p.numel() for p in model.parameters())
        flops = count_flops(model, cfg, device)
        print(json.dumps({"model": name, "params": params, **flops}), flush=True)
        run_cfg = dict(cfg, lr=learning_rate(cfg, name))
        result = train_model(model, train_set, val_set, run_cfg, device, lambda row, name=name: print(json.dumps({"model": name, **row}), flush=True))
        save_checkpoint(cfg, name, model, model_cfg, cipher_vocab, plain_vocab, {"params": params, **flops, "best_step": result["best_step"], "best_val_loss": result["best_val_loss"]})
        write_json(Path(cfg["results_dir"]) / f"curves_{name}.json", {"model": name, "config": cfg, "params": params, **result})
        entry = {"params": params, **flops, "lr": run_cfg["lr"], "best_step": result["best_step"], "best_val_loss": result["best_val_loss"], "train_minutes": result["train_minutes"]}
        if cfg["final_eval"]:
            for split in cfg["eval_splits"]:
                records, metrics = evaluate_split(
                    model, cipher_vocab, plain_vocab, cfg, device, split, cfg.get("eval_max_docs"), cfg.get("eval_max_chunks"), cfg.get("eval_max_minutes")
                )
                write_jsonl(Path(cfg["output_dir"]) / name / f"{split}_decrypt.jsonl", records)
                entry[split] = metrics
                print(json.dumps({"model": name, "split": split, **metrics}), flush=True)
        summary[name] = entry
        write_json(Path(cfg["results_dir"]) / f"summary_{name}.json", entry)
    return summary


def load_model(cfg, name, source, device):
    payload, cipher_vocab, plain_vocab = load_checkpoint(fetch_checkpoint(cfg, name, source))
    model = build_model(payload["model_cfg"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, cipher_vocab, plain_vocab


def run_evaluate(cfg, names, split, source, device):
    summary = {}
    for name in names:
        model, cipher_vocab, plain_vocab = load_model(cfg, name, source, device)
        records, metrics = evaluate_split(model, cipher_vocab, plain_vocab, cfg, device, split)
        write_jsonl(Path(cfg["output_dir"]) / name / f"{split}_decrypt.jsonl", records)
        if name == cfg["inference_model"]:
            write_jsonl(Path(cfg["output_dir"]) / f"{split}_decrypt.jsonl", records)
        summary[name] = {"params": sum(p.numel() for p in model.parameters()), **count_flops(model, cfg, device), split: metrics}
        print(json.dumps({"model": name, "split": split, "metrics": metrics}), flush=True)
    write_json(Path(cfg["results_dir"]) / f"eval_{split}.json", summary)


def run_heldout(cfg, source, device):
    model, cipher_vocab, plain_vocab = load_model(cfg, cfg["inference_model"], source, device)
    cipher_path, plain_path = heldout_paths(cfg["heldout_path"])
    corpus = PairCorpus(cipher_path, plain_path, drop_misaligned=False)
    records, metrics = decrypt_corpus(model, corpus, cipher_vocab, plain_vocab, cfg, device)
    write_jsonl(Path(cfg["output_dir"]) / f"heldout_result_{cfg['roll_number']}.jsonl", records)
    if metrics is not None:
        print(json.dumps({"model": cfg["inference_model"], "split": "heldout", **metrics}), flush=True)


def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)
    names = model_names(cfg, args.model)
    if args.command == "train":
        summary = run_train(cfg, names, device)
        write_json(Path(cfg["results_dir"]) / "summary.json", summary)
    elif args.command == "evaluate":
        run_evaluate(cfg, names, args.split, args.source, device)
    elif args.command == "heldout":
        run_heldout(cfg, args.source, device)
    else:
        push_checkpoints(cfg, names)


if __name__ == "__main__":
    main()
