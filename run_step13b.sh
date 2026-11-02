#!/bin/zsh
# Step 13b: level-dependent faults on the retrained ResNet-18 (3 sigmas x 3 conditions x 5 seeds = 45, plus the
# fault-free reference); no repair and BatchNorm recalibration. Evaluation only. Stops at the first failure.
set -euo pipefail
cd "$(dirname "$0")"
echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) starting: resnet18_cifar10_level_faults ==="
.venv/bin/python scripts/run_level_fault_sweep.py --config configs/resnet18_cifar10_level_faults.json
echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) finished: resnet18_cifar10_level_faults ==="
echo "=== ALL STEP 13B RUNS COMPLETE ==="
