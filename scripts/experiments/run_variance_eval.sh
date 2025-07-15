#!/usr/bin/env bash
set -euo pipefail

# ensure script runs from project root
SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$ROOT_DIR"

# base results directory and parameters
ROOT="./results/cifar10_resnet18_32x32_base_e100_lr0.1_default"
POST="variance"
PLOT=false

# define hyperparameter sweeps
NOISES=(0.00784313725 0.0156862745 0.031372549019608 0.06274509803921569)
NUM_SAMPLES=(10 100)

# paths to config
CFG_PATH="configs/postprocessors/variance.yml"
CFG_BAK="$CFG_PATH.bak"

# backup original config
cp "$CFG_PATH" "$CFG_BAK"

# output directory for CSVs
OUTPUT_DIR="scripts/experiments/outputs"
mkdir -p "$OUTPUT_DIR"

# define master CSV for aggregation
MASTER_CSV="$OUTPUT_DIR/${POST}_sweep.csv"
# initialize (or truncate) master CSV
> "$MASTER_CSV"

for noise in "${NOISES[@]}"; do
  for samples in "${NUM_SAMPLES[@]}"; do
    echo "Running variance eval: noise=${noise}, samples=${samples}"
    # override YAML config using yq if available, else sed
    if command -v yq >/dev/null 2>&1; then
      yq e ".postprocessor.postprocessor_args.noise_magnitude = ${noise} |\
             .postprocessor.postprocessor_args.num_samples = ${samples}" \
             "$CFG_BAK" > "$CFG_PATH"
    else
      sed -E -i "s/(noise_magnitude:).*/\1 ${noise}/" "$CFG_PATH"
      sed -E -i "s/(num_samples:).*/\1 ${samples}/" "$CFG_PATH"
    fi

    # run evaluation
    python scripts/eval_ood.py --root "$ROOT" --save-csv --postprocessor "$POST" --plot-score "$PLOT"

    # append this run to master CSV
    echo "noise=${noise},samples=${samples}" >> "$MASTER_CSV"
    cat "$ROOT/ood/${POST}.csv" >> "$MASTER_CSV"
    echo "" >> "$MASTER_CSV"

    echo "Completed run: noise=${noise}, samples=${samples}"
    echo "----------------------------------------"
  done
done

# restore original config
mv "$CFG_BAK" "$CFG_PATH"
echo "All runs complete. CSVs are in $OUTPUT_DIR." 
echo "----------------------------------------"