set -euo pipefail
cd "$(dirname "$0")/../.."
uv run python -m src.part2.log_wandb --project anlp_a3_part2 "$@"
