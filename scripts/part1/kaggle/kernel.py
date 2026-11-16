import os
import shutil
import subprocess
import sys
from pathlib import Path

INPUT = Path("/kaggle/input")
REPO = Path("/tmp/repo")
WORKING = Path("/kaggle/working")
COMMANDS = [["src.part1.probe"]]
HELDOUT_FROM = None


def locate_code():
    configs = [path for path in INPUT.rglob("pyproject.toml") if (path.parent / "src").exists()]
    if configs:
        shutil.copytree(configs[0].parent, REPO, dirs_exist_ok=True)
        return
    shutil.unpack_archive(next(INPUT.rglob("code.zip")), REPO)


def link_data():
    source = next(path.parent.parent for path in INPUT.rglob("train/cipher.jsonl"))
    target = REPO / "data" / "part1"
    target.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        if not (target / split).exists():
            os.symlink(source / split, target / split)
    if HELDOUT_FROM is not None and not (target / "heldout").exists():
        os.symlink(source / HELDOUT_FROM, target / "heldout")


def main():
    locate_code()
    link_data()
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rapidfuzz", "huggingface_hub"], check=True)
    for command in COMMANDS:
        subprocess.run([sys.executable, "-m", *command], cwd=REPO, check=True)
    for folder in ("output/part1", "results/part1", "models/part1"):
        if (REPO / folder).exists():
            shutil.copytree(REPO / folder, WORKING / folder, dirs_exist_ok=True)


if __name__ == "__main__":
    main()
