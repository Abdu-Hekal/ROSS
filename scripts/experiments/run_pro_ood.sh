#!/usr/bin/env bash
set -euo pipefail

# ensure script runs from project root
SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$ROOT_DIR"

POST="pro"
PY_SCRIPT="scripts/eval_ood.py"
IMG_SCRIPT="scripts/eval_ood_imagenet.py"
CONFIG="configs/postprocessors/pro.yml"
CONFIG_BAK="${CONFIG}.bak"
BENCHES=(cifar10 cifar100 imagenet200 imagenet1k)
BASE_PPS=(fdbd)
SAVE_CSV="--save-csv"
PLOT_SCORE=false

# backup original config
cp "$CONFIG" "$CONFIG_BAK"

OUTPUT_BASE="scripts/experiments/outputs/pro_ood"
mkdir -p "$OUTPUT_BASE"

for bench in "${BENCHES[@]}"; do
  echo "Benchmark: $bench"
  if [[ "$bench" == "imagenet1k" ]]; then
    SCRIPT="$IMG_SCRIPT"
    ROOT=$(ls -d results/imagenet_resnet50_tvsv*_base_default | head -n1)
    CMD="python $SCRIPT --root $ROOT --tvs-pretrained --postprocessor $POST --save-csv"
    SRC_DIR="$ROOT/ood"
  else
    SCRIPT="$PY_SCRIPT"
    ROOT=$(ls -d results/${bench}_* | head -n1)
    CMD="python $SCRIPT --root $ROOT --id-data $bench --postprocessor $POST $SAVE_CSV --plot-score $PLOT_SCORE"
    SRC_DIR="$ROOT/ood"
  fi

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
echo "PRO OOD sweep complete." 