set -euo pipefail
cd "$(dirname "$0")/../.."
: "${KAGGLE_USERNAME:?export KAGGLE_USERNAME first}"
STAGE=logs/part2/kaggle_stage
DATASET="$KAGGLE_USERNAME/anlp-a3-code"
rm -rf "$STAGE"
mkdir -p "$STAGE/code" "$STAGE/kernel"
zip -qr "$STAGE/code/code.zip" src config pyproject.toml results/part2/results_template.json -x "*/__pycache__/*"
sed "s/KAGGLE_USERNAME/$KAGGLE_USERNAME/g" scripts/part2/kaggle/dataset-metadata.json > "$STAGE/code/dataset-metadata.json"
sed "s/KAGGLE_USERNAME/$KAGGLE_USERNAME/g" scripts/part2/kaggle/kernel-metadata.json > "$STAGE/kernel/kernel-metadata.json"
MODULE="${1:-src.part2.main}"
CONFIG="${2:-config/part2/gnn.yaml}"
sed -e "s|^MODULE = .*|MODULE = \"$MODULE\"|" -e "s|^CONFIG = .*|CONFIG = \"$CONFIG\"|" scripts/part2/kaggle/kernel.py > "$STAGE/kernel/kernel.py"
if kaggle datasets status "$DATASET" >/dev/null 2>&1; then
  kaggle datasets version -p "$STAGE/code" -m "code update"
else
  kaggle datasets create -p "$STAGE/code"
fi
until kaggle datasets status "$DATASET" 2>/dev/null | grep -qi ready; do
  sleep 10
done
kaggle kernels push -p "$STAGE/kernel"
