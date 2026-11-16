import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

from src.part2.log_wandb import load_env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoints", default="models/part2")
    return parser.parse_args()


def main():
    args = parse_args()
    load_env(Path(".env"))
    api = HfApi(token=os.environ.get("HUGGINGFACE_KEY"))
    api.create_repo(args.repo, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(repo_id=args.repo, folder_path=args.checkpoints, path_in_repo="part2", allow_patterns=["*.pt"])
    for name in ("config/part2/gnn.yaml", "results/part2/summary.json"):
        if Path(name).exists():
            api.upload_file(path_or_fileobj=name, path_in_repo=f"part2/{Path(name).name}", repo_id=args.repo)


if __name__ == "__main__":
    main()
