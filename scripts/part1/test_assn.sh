set -euo pipefail
cd "$(dirname "$0")/../.."
if [ "${1:-}" = "--heldout" ]; then
  shift
  uv run python -m src.part1.main heldout --source hf "$@"
else
  uv run python -m src.part1.main evaluate --model all --split test --source hf "$@"
fi
