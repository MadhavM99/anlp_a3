set -euo pipefail
cd "$(dirname "$0")/../.."
: "${KAGGLE_USERNAME:?export KAGGLE_USERNAME first}"
KERNEL="$KAGGLE_USERNAME/anlp-a3-part1"
OUT=logs/part1/kaggle_output
kaggle kernels status "$KERNEL"
rm -rf "$OUT"
mkdir -p "$OUT"
kaggle kernels output "$KERNEL" -p "$OUT"
for folder in output/part1 results/part1 models/part1; do
  mkdir -p "$folder"
  if [ -d "$OUT/$folder" ]; then
    cp -R "$OUT/$folder/." "$folder/"
  fi
done
