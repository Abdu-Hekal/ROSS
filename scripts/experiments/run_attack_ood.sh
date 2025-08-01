#!/usr/bin/env bash
set -euo pipefail

# ensure script runs from project root
SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$ROOT_DIR"

ATTACK_SCRIPT="scripts/attack_ood.py"
ID="cifar10"
ROOT="results/cifar10_resnet18_32x32_base_e100_lr0.1_default"
SAVE_CSV="--save-csv"
BATCH_SIZE=128

POSTS=() #(msp ebo gen fdbd)
ATTACK_METHODS=(LinfPGD) #(FGSM LinfPGD DeepFool)
EPSILONS=(0.007843137 0.0156862745) #0.031372549 0.0627410098
OBJECTIVES=(min max)

OUTPUT_BASE="scripts/experiments/outputs/attack_ood"
mkdir -p "$OUTPUT_BASE"

# base postprocessor attacks
for pp in "${POSTS[@]}"; do
  for attack in "${ATTACK_METHODS[@]}"; do
    for eps in "${EPSILONS[@]}"; do
      for obj in "${OBJECTIVES[@]}"; do
        echo "Attacking PP=$pp, attack=$attack, eps=$eps, obj=$obj"
        rm -f "$ROOT/attack_ood/${pp}_${attack}.csv"
        python "$ATTACK_SCRIPT" --root "$ROOT" --id-data "$ID" \
          --postprocessor "$pp" --attack-method "$attack" --eps "$eps" \
          --steps 40 --ood-objective "$obj" $SAVE_CSV --batch-size "$BATCH_SIZE" --reuse-attack
        mv "$ROOT/attack_ood/${pp}_${attack}.csv" \
           "$OUTPUT_BASE/${pp}_${attack}_eps${eps}_${obj}.csv"
      done
done
done
done

# composite attacks: backup configs
CONFIG_PRO="configs/postprocessors/pro.yml"
CONFIG_PRO_BAK="${CONFIG_PRO}.bak"
CONFIG_VAR="configs/postprocessors/variance.yml"
CONFIG_VAR_BAK="${CONFIG_VAR}.bak"
cp "$CONFIG_PRO" "$CONFIG_PRO_BAK"
cp "$CONFIG_VAR" "$CONFIG_VAR_BAK"

# PRO-FDBD attacks
# echo "Running PRO-FDBD attacks"
# if command -v yq >/dev/null 2>&1; then
#   yq e ".postprocessor.postprocessor_args.score_postprocessor = \"fdbd\"" \
#      "$CONFIG_PRO_BAK" > "$CONFIG_PRO"
# else
#   sed -E -i "s/(score_postprocessor:).*/\1 fdbd/" "$CONFIG_PRO"
# fi
# for attack in "${ATTACK_METHODS[@]}"; do
#   for eps in "${EPSILONS[@]}"; do
#     for obj in "${OBJECTIVES[@]}"; do
#       echo "PRO-FDBD, attack=$attack, eps=$eps, obj=$obj"
#       rm -f "$ROOT/attack_ood/pro_${attack}.csv"
#       python "$ATTACK_SCRIPT" --root "$ROOT" --id-data "$ID" \
#         --postprocessor pro --attack-method "$attack" --eps "$eps" \
#         --steps 40 --ood-objective "$obj" --attack-base-pp $SAVE_CSV --batch-size "$BATCH_SIZE" --reuse-attack
#       mv "$ROOT/attack_ood/pro_${attack}.csv" \
#          "$OUTPUT_BASE/pro_fdbd_${attack}_eps${eps}_${obj}.csv"
#     done
# done
# done

# Variance attacks
echo "Running Variance attacks"
BASE_PPS=(fdbd)
NOISES=(0.25 0.5)
for base_pp in "${BASE_PPS[@]}"; do
  for noise in "${NOISES[@]}"; do
    if command -v yq >/dev/null 2>&1; then
      yq e ".postprocessor.postprocessor_args.score_postprocessor = \"${base_pp}\" | \
             .postprocessor.postprocessor_args.noise_magnitude = ${noise}" \
             "$CONFIG_VAR_BAK" > "$CONFIG_VAR"
    else
      sed -E -i "s/(score_postprocessor:).*/\1 ${base_pp}/" "$CONFIG_VAR"
      sed -E -i "s/(noise_magnitude:).*/\1 ${noise}/" "$CONFIG_VAR"
    fi
    for attack in "${ATTACK_METHODS[@]}"; do
      for eps in "${EPSILONS[@]}"; do
        for obj in "${OBJECTIVES[@]}"; do
          echo "Variance-${base_pp}, noise=$noise, attack=$attack, eps=$eps, obj=$obj"
          rm -f "$ROOT/attack_ood/variance_${attack}.csv"
          python "$ATTACK_SCRIPT" --root "$ROOT" --id-data "$ID" \
            --postprocessor variance --attack-method "$attack" --eps "$eps" \
            --steps 40 --ood-objective "$obj" --attack-base-pp $SAVE_CSV --batch-size "$BATCH_SIZE" --reuse-attack
          mv "$ROOT/attack_ood/variance_${attack}.csv" \
             "$OUTPUT_BASE/variance_${base_pp}_noise${noise}_${attack}_eps${eps}_${obj}.csv"
        done
      done
    done
  done

done

# ODIN-base attacks
# echo "Running ODIN base-PP attacks"
# for attack in "${ATTACK_METHODS[@]}"; do
#   for eps in "${EPSILONS[@]}"; do
#     for obj in "${OBJECTIVES[@]}"; do
#       echo "ODIN, attack=$attack, eps=$eps, obj=$obj"
#       rm -f "$ROOT/attack_ood/odin_${attack}.csv"
#       python "$ATTACK_SCRIPT" --root "$ROOT" --id-data "$ID" \
#         --postprocessor odin --attack-method "$attack" --eps "$eps" \
#         --steps 40 --ood-objective "$obj" --attack-base-pp $SAVE_CSV --batch-size "$BATCH_SIZE" --reuse-attack
#       mv "$ROOT/attack_ood/odin_${attack}.csv" \
#          "$OUTPUT_BASE/odin_${attack}_eps${eps}_${obj}.csv"
#     done
#   done
# done

# restore configs
mv "$CONFIG_PRO_BAK" "$CONFIG_PRO"
mv "$CONFIG_VAR_BAK" "$CONFIG_VAR"
echo "Attack OOD sweep complete." 