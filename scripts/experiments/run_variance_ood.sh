#!/usr/bin/env bash
set -euo pipefail

# ensure script runs from project root
SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$ROOT_DIR"

POST="variance"
PY_SCRIPT="scripts/eval_ood.py"
IMG_SCRIPT="scripts/eval_ood_imagenet.py"
CONFIG="configs/postprocessors/variance.yml"
CONFIG_BAK="${CONFIG}.bak"
BENCHES=(cifar10 cifar100)
BASE_PPS=(gen msp)
SAVE_CSV="--save-csv"
PLOT_SCORE=false

# backup original config
cp "$CONFIG" "$CONFIG_BAK"

OUTPUT_BASE="scripts/experiments/outputs/variance_ood"
mkdir -p "$OUTPUT_BASE"

for bench in "${BENCHES[@]}"; do
  echo "Benchmark: $bench"
  SCRIPT="$PY_SCRIPT"
  ROOT=$(ls -d results/${bench}_* | head -n1)
  CMD="python $SCRIPT --root $ROOT --id-data $bench --postprocessor $POST $SAVE_CSV --plot-score $PLOT_SCORE --batch-size 64"
  SRC_DIR="$ROOT/ood"

  OUT_DIR="$OUTPUT_BASE/$bench"
  mkdir -p "$OUT_DIR"

  for pp in "${BASE_PPS[@]}"; do
    echo "  Base PP: $pp"
    # patch score_postprocessor
    if command -v yq >/dev/null 2>&1; then
      yq e ".postprocessor.postprocessor_args.score_postprocessor = \"$pp\"" "$CONFIG_BAK" > "$CONFIG"
    else
      sed -E -i "s/(score_postprocessor:).*/\1 $pp/" "$CONFIG"
    fi

    # run evaluation
    eval $CMD

    # move csv
    mv "$SRC_DIR/${POST}.csv" "$OUT_DIR/${POST}_$pp.csv"
  done
done

# restore original config
mv "$CONFIG_BAK" "$CONFIG"
echo "Variance OOD sweep complete." 