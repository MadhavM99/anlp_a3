set -euo pipefail
cd "$(dirname "$0")/../.."
: "${KAGGLE_USERNAME:?export KAGGLE_USERNAME first}"
if [ -z "${COMMANDS:-}" ]; then
  COMMANDS=$(python3 -I -c 'import json, sys; print(json.dumps([sys.argv[1:] or ["src.part1.main"]]))' "$@")
fi
HELDOUT_FROM="${HELDOUT_FROM:-None}"
if [ "$HELDOUT_FROM" != "None" ]; then
  HELDOUT_FROM="\"$HELDOUT_FROM\""
fi
STAGE=logs/part1/kaggle_stage
DATASET="$KAGGLE_USERNAME/anlp-a3-code"
rm -rf "$STAGE"
mkdir -p "$STAGE/code" "$STAGE/kernel"
zip -qr "$STAGE/code/code.zip" src config pyproject.toml results/part2/results_template.json -x "*/__pycache__/*"
sed "s/KAGGLE_USERNAME/$KAGGLE_USERNAME/g" scripts/part2/kaggle/dataset-metadata.json > "$STAGE/code/dataset-metadata.json"
sed -e "s/KAGGLE_USERNAME/$KAGGLE_USERNAME/g" -e "s/\"enable_gpu\": true/\"enable_gpu\": ${ENABLE_GPU:-true}/" scripts/part1/kaggle/kernel-metadata.json > "$STAGE/kernel/kernel-metadata.json"
sed -e "s|^COMMANDS = .*|COMMANDS = $COMMANDS|" -e "s|^HELDOUT_FROM = .*|HELDOUT_FROM = $HELDOUT_FROM|" scripts/part1/kaggle/kernel.py > "$STAGE/kernel/kernel.py"
kaggle datasets version -p "$STAGE/code" -m "code update"
until kaggle datasets status "$DATASET" 2>/dev/null | grep -qi ready; do
  sleep 10
done
kaggle kernels push -p "$STAGE/kernel"
