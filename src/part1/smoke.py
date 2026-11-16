import json
import time

import torch

from src.part1.main import load_config, run_train, write_json
from src.part1.models import build_model
from src.part1.models.deltanet import delta_rule_chunkwise


def delta_rule_recurrent(q, k, v, beta):
    state = q.new_zeros(q.size(0), q.size(1), q.size(-1), v.size(-1))
    outputs = []
    for t in range(q.size(2)):
        k_t, v_t, b_t = k[:, :, t], v[:, :, t], beta[:, :, t, None]
        old = torch.einsum("bhk,bhkv->bhv", k_t, state)
        state = state + torch.einsum("bhk,bhv->bhkv", k_t, b_t * (v_t - old))
        outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, :, t], state))
    return torch.stack(outputs, dim=2)


def check_delta_rule(device):
    errors = {}
    for length in (5, 32, 68):
        q = torch.nn.functional.normalize(torch.randn(2, 4, length, 16, device=device), dim=-1)
        k = torch.nn.functional.normalize(torch.randn(2, 4, length, 16, device=device), dim=-1)
        v = torch.randn(2, 4, length, 16, device=device)
        beta = torch.rand(2, 4, length, device=device)
        errors[length] = (delta_rule_chunkwise(q, k, v, beta, 32) - delta_rule_recurrent(q, k, v, beta)).abs().max().item()
    return errors


def check_causality(cfg, device):
    report = {}
    for name in cfg["models"]:
        model_cfg = dict(cfg["model"], name=name, input_vocab=65, output_vocab=27)
        model = build_model(model_cfg).to(device).eval()
        x = torch.randint(2, 65, (2, 68), device=device)
        y = x.clone()
        y[:, 40] = (y[:, 40] + 1) % 63 + 2
        with torch.no_grad():
            leak = (model(x)[:, :40] - model(y)[:, :40]).abs().max().item()
            changed = (model(x)[:, 40:] - model(y)[:, 40:]).abs().max().item()
        report[name] = {"params": sum(p.numel() for p in model.parameters()), "leak_before_40": leak, "change_from_40": changed, "layout": model.kinds}
    return report


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config("config/part1/base.yaml", [])
    torch.manual_seed(0)
    checks = {"delta_rule_max_abs_error": check_delta_rule(device), "models": check_causality(cfg, device)}
    print(json.dumps(checks, indent=2), flush=True)
    cfg.update(
        max_steps=20, log_interval=5, eval_interval=10, eval_limit=5, max_docs=2000, val_max_docs=100,
        eval_max_docs=20, results_dir="results/part1/smoke", output_dir="output/part1/smoke", checkpoint_dir="models/part1/smoke",
    )
    started = time.time()
    summary = run_train(cfg, cfg["models"], device)
    checks["smoke_minutes"] = (time.time() - started) / 60
    checks["smoke_summary"] = summary
    write_json("results/part1/smoke/checks.json", checks)
    print(json.dumps({"smoke_minutes": checks["smoke_minutes"]}), flush=True)


if __name__ == "__main__":
    main()
