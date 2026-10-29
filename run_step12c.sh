#!/bin/zsh
# Step 12c: TinyBERT LoRA on the codebook-retrained Condition B baseline, two
# adapter placements: 48 runs on MPS, then 54 transfer evaluations, as one
# declared matrix. Stops at the first failure.
set -euo pipefail
cd "$(dirname "$0")"
echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) starting matrix: tinybert_sst2_lora12c_matrix ==="
.venv/bin/python scripts/run_tinybert_retrained_recovery_matrix.py --config configs/tinybert_sst2_lora12c_matrix.json
echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) finished matrix: tinybert_sst2_lora12c_matrix ==="
echo "=== ALL STEP 12C RUNS COMPLETE ==="
