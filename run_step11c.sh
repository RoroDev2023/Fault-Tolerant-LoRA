#!/bin/zsh
# Step 11c: BatchNorm recalibration on the retrained encoding, then the 43 LoRA
# runs as five sequential matrices. Stops at the first failure.
set -euo pipefail
cd "$(dirname "$0")"
echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) starting BatchNorm recalibration ==="
.venv/bin/python scripts/run_bn_recalibration.py --config configs/resnet18_cifar10_retrained_bn_recalibration.json
echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) finished BatchNorm recalibration ==="
for name in controls p001 p005 p010 p020; do
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) starting matrix: retrained_$name ==="
  .venv/bin/python scripts/run_lora_matrix.py --config "configs/resnet18_cifar10_lora_retrained_${name}.json"
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) finished matrix: retrained_$name ==="
done
echo "=== ALL STEP 11C RUNS COMPLETE ==="
