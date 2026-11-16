set -euo pipefail
cd "$(dirname "$0")/../.."
uv run python -m src.part2.main "$@"
