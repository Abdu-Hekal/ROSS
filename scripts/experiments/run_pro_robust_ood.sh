#!/usr/bin/env bash
set -euo pipefail

# ensure script runs from project root
SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$ROOT_DIR"

POST="pro"
SCRIPT="scripts/eval_ood_aarobust.py"
CONFIG="configs/postprocessors/pro.yml"
CONFIG_BAK="${CONFIG}.bak"
BENCHES=(cifar10 cifar100 imagenet200)
BASE_PPS=fdbd
MODELIDS=(
  'Diffenderfer2021Winning_LRR'
)

# backup original config
cp "$CONFIG" "$CONFIG_BAK"

OUTPUT_BASE="scripts/experiments/outputs/pro_robust_ood"
mkdir -p "$OUTPUT_BASE"

for bench in "${BENCHES[@]}"; do
  echo "Benchmark: $bench"
  if [[ "$bench" == "imagenet200" ]]; then
    ID="imagenet"
  else
    ID="$bench"
  fi
  OUT_DIR="$OUTPUT_BASE/$bench"
  mkdir -p "$OUT_DIR"

  for modelid in "${MODELIDS[@]}"; do
    echo "  Model ID: $modelid"
    OUT_DIR="$OUTPUT_BASE/$bench/$modelid"
    mkdir -p "$OUT_DIR"

    for pp in "${BASE_PPS[@]}"; do
      echo "    Base PP: $pp"
      # patch score_postprocessor
      if command -v yq >/dev/null 2>&1; then
        yq e ".postprocessor.postprocessor_args.score_postprocessor = \"$pp\"" "$CONFIG_BAK" > "$CONFIG"
      else
        sed -E -i "s/(score_postprocessor:).*/\1 $pp/" "$CONFIG"
      fi

      # run robust evaluation
      python "$SCRIPT" --id-data "$ID" --modelid "$modelid" --postprocessor "$POST" --save-csv --overwrite --savekeyword "$pp"

      # move csv
      mv "results/robustmodels/$ID/corruptions/$modelid/ood/${POST}_$pp.csv" "$OUT_DIR/${POST}_$pp.csv"
    done
  done
done

# restore original config
mv "$CONFIG_BAK" "$CONFIG"
echo "PRO Robust OOD sweep complete." 