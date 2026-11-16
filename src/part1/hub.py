import json
import os
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download

from src.part1.data import Vocab


def load_env(path=Path(".env")):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def checkpoint_path(cfg, name):
    return Path(cfg["checkpoint_dir"]) / name / "model.pt"


def save_checkpoint(cfg, name, model, model_cfg, cipher_vocab, plain_vocab, extra):
    path = checkpoint_path(cfg, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "model_cfg": model_cfg,
        "cipher_vocab": cipher_vocab.to_list(),
        "plain_vocab": plain_vocab.to_list(),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    torch.save(payload, path)
    (path.parent / "config.json").write_text(json.dumps({"name": name, "model_cfg": model_cfg, **extra}, indent=2))
    return path


def fetch_checkpoint(cfg, name, source):
    if source == "local":
        return checkpoint_path(cfg, name)
    load_env()
    return Path(hf_hub_download(repo_id=cfg["hf_repo"], filename=f"{name}/model.pt", token=os.environ.get("HUGGINGFACE_KEY")))


def load_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload, Vocab.from_list(payload["cipher_vocab"]), Vocab.from_list(payload["plain_vocab"])


def push_checkpoints(cfg, names):
    load_env()
    api = HfApi(token=os.environ.get("HUGGINGFACE_KEY"))
    api.create_repo(cfg["hf_repo"], repo_type="model", private=False, exist_ok=True)
    for name in names:
        folder = checkpoint_path(cfg, name).parent
        api.upload_folder(repo_id=cfg["hf_repo"], folder_path=str(folder), path_in_repo=name)
