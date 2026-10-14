#!/bin/zsh
# Part 2: the 32 LoRA control runs left after the interrupted controls matrix,
# as four sequential matrices. Stops at the first failure.
set -euo pipefail
cd "$(dirname "$0")"
for name in controls_b faulty_p001 faulty_p005 faulty_p010; do
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) starting matrix: $name ==="
  .venv/bin/python scripts/run_lora_matrix.py --config "configs/resnet18_cifar10_lora_${name}.json"
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) finished matrix: $name ==="
done
echo "=== ALL PART 2 MATRICES COMPLETE ==="
