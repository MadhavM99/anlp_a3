set -euo pipefail
cd "$(dirname "$0")/../.."
uv run python -m src.part1.main train --model all "$@"
