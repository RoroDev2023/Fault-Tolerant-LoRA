#!/bin/zsh
# Part 7: the TinyBERT LoRA recovery matrix (50 runs on MPS), then its 63
# transfer evaluations, as one declared matrix. Stops at the first failure.
set -euo pipefail
cd "$(dirname "$0")"
echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) starting matrix: tinybert_sst2_lora_matrix_part7_proposed ==="
.venv/bin/python scripts/run_tinybert_recovery_matrix.py --config configs/tinybert_sst2_lora_matrix_part7_proposed.json
echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) finished matrix: tinybert_sst2_lora_matrix_part7_proposed ==="
echo "=== ALL PART 7 MATRICES COMPLETE ==="
