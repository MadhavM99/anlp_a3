import copy
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.part2.data import make_loaders
from src.part2.models import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(labels, preds, average):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=average, zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def run_epoch(model, loader, device, average, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total_graphs = 0.0, 0
    preds, labels = [], []
    with torch.set_grad_enabled(training):
        for data in loader:
            data = data.to(device)
            logits = model(data.x, data.edge_index, data.batch, data.ptr)
            loss = F.cross_entropy(logits, data.y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * data.num_graphs
            total_graphs += data.num_graphs
            preds.append(logits.argmax(dim=-1).cpu())
            labels.append(data.y.cpu())
    metrics = compute_metrics(torch.cat(labels).numpy(), torch.cat(preds).numpy(), average)
    metrics["loss"] = total_loss / total_graphs
    return metrics


def train_run(model_name, feature, seed, splits, cfg, device, evaluate_test=True):
    set_seed(seed)
    loaders = make_loaders(splits, cfg["batch_size"], torch.Generator().manual_seed(seed))
    model = build_model(model_name, splits["train"].num_features, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    best_f1, best_epoch, best_state = -1.0, 0, None
    curve = []
    for epoch in range(1, cfg["epochs"] + 1):
        train_metrics = run_epoch(model, loaders["train"], device, cfg["average"], optimizer)
        val_metrics = run_epoch(model, loaders["val"], device, cfg["average"])
        row = {"epoch": epoch}
        row.update({f"train/{k}": v for k, v in train_metrics.items()})
        row.update({f"val/{k}": v for k, v in val_metrics.items()})
        curve.append(row)
        if val_metrics["f1"] > best_f1:
            best_f1, best_epoch = val_metrics["f1"], epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    test_metrics = None
    if evaluate_test:
        test_metrics = run_epoch(model, loaders["test"], device, cfg["average"])
        test_metrics.pop("loss")
    return {
        "model": model_name,
        "feature": feature,
        "seed": seed,
        "params": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "test": test_metrics,
        "curve": curve,
        "state": best_state,
    }
