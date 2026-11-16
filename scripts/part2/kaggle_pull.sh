set -euo pipefail
cd "$(dirname "$0")/../.."
: "${KAGGLE_USERNAME:?export KAGGLE_USERNAME first}"
KERNEL="$KAGGLE_USERNAME/anlp-a3-part2"
OUT=logs/part2/kaggle_output
kaggle kernels status "$KERNEL"
rm -rf "$OUT"
mkdir -p "$OUT"
kaggle kernels output "$KERNEL" -p "$OUT"
for folder in output/part2 results/part2 models/part2; do
  mkdir -p "$folder"
  if [ -d "$OUT/$folder" ]; then
    cp -R "$OUT/$folder/." "$folder/"
  fi
done
