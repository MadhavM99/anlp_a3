import shutil
import subprocess
import sys
from pathlib import Path

INPUT = Path("/kaggle/input")
REPO = Path("/tmp/repo")
WORKING = Path("/kaggle/working")
MODULE = "src.part2.main"
CONFIG = "config/part2/gnn.yaml"


def locate_code():
    configs = list(INPUT.rglob("config/part2/gnn.yaml"))
    if configs:
        shutil.copytree(configs[0].parents[2], REPO, dirs_exist_ok=True)
        return
    archive = next(INPUT.rglob("code.zip"))
    shutil.unpack_archive(archive, REPO)


def main():
    locate_code()
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch_geometric"], check=True)
    subprocess.run([sys.executable, "-m", MODULE, "--config", CONFIG], cwd=REPO, check=True)
    for folder in ("output/part2", "results/part2", "models/part2"):
        if (REPO / folder).exists():
            shutil.copytree(REPO / folder, WORKING / folder, dirs_exist_ok=True)


if __name__ == "__main__":
    main()
