import copy
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.part1.data import IGNORE


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_schedule(step, warmup, total):
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def parameter_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or getattr(parameter, "no_weight_decay", False) or "norm" in name or "embedding" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def batch_loss(model, x, y, label_smoothing, amp, device):
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
        logits = model(x)
    logits = logits.float()
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=IGNORE, label_smoothing=label_smoothing)
    mask = y != IGNORE
    correct = ((logits.argmax(dim=-1) == y) & mask).sum().item()
    return loss, correct, mask.sum().item()


@torch.no_grad()
def evaluate_loss(model, loader, cfg, device, limit):
    model.eval()
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    for index, (x, y) in enumerate(loader):
        if index >= limit:
            break
        x, y = x.to(device), y.to(device)
        loss, correct, tokens = batch_loss(model, x, y, 0.0, cfg["amp"], device)
        total_loss += loss.item() * tokens
        total_correct += correct
        total_tokens += tokens
    model.train()
    return {"loss": total_loss / max(1, total_tokens), "char_accuracy": total_correct / max(1, total_tokens)}


def infinite(loader):
    while True:
        yield from loader


def train_model(model, train_set, val_set, cfg, device, log):
    set_seed(cfg["seed"])
    generator = torch.Generator().manual_seed(cfg["seed"])
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, generator=generator, num_workers=cfg["workers"], drop_last=True)
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["workers"])
    optimizer = torch.optim.AdamW(parameter_groups(model, cfg["weight_decay"]), lr=cfg["lr"], betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: cosine_schedule(step, cfg["warmup_steps"], cfg["max_steps"]))
    scaler = torch.amp.GradScaler(device.type, enabled=cfg["amp"] and device.type == "cuda")
    model.train()
    curve, best = [], {"val_loss": float("inf"), "step": 0, "state": None}
    window_loss, window_correct, window_tokens = 0.0, 0, 0
    started = time.time()
    for step, (x, y) in enumerate(infinite(train_loader), start=1):
        if step > cfg["max_steps"]:
            break
        x, y = x.to(device), y.to(device)
        loss, correct, tokens = batch_loss(model, x, y, cfg["label_smoothing"], cfg["amp"], device)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        window_loss += loss.item() * tokens
        window_correct += correct
        window_tokens += tokens
        if step % cfg["log_interval"] == 0 or step == cfg["max_steps"]:
            row = {
                "step": step,
                "train/loss": window_loss / max(1, window_tokens),
                "train/char_accuracy": window_correct / max(1, window_tokens),
                "train/lr": scheduler.get_last_lr()[0],
                "train/grad_norm": float(grad_norm),
                "train/elapsed_min": (time.time() - started) / 60,
            }
            window_loss, window_correct, window_tokens = 0.0, 0, 0
            if step % cfg["eval_interval"] == 0 or step == cfg["max_steps"]:
                val = evaluate_loss(model, val_loader, cfg, device, cfg["eval_limit"])
                row.update({f"val/{k}": v for k, v in val.items()})
                if val["loss"] < best["val_loss"]:
                    best = {"val_loss": val["loss"], "step": step, "state": copy.deepcopy(model.state_dict())}
                row["val/best_loss"] = best["val_loss"]
            curve.append(row)
            log(row)
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return {"curve": curve, "best_step": best["step"], "best_val_loss": best["val_loss"], "train_minutes": (time.time() - started) / 60}
