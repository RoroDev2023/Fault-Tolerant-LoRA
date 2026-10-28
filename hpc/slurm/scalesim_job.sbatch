#!/usr/bin/env bash
# Submit from the repository root. SCALE-Sim is CPU-based in this workflow;
# provide NYU-specific account, partition, time, memory, CPU, and log options.
#SBATCH --job-name=fault-lora-scalesim
#SBATCH --ntasks=1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

repo_root="${FAULT_LORA_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
repo_root="$(cd "$repo_root" && pwd)"
cd "$repo_root"

if [[ -n "${SCALESIM_ENV_ACTIVATE:-}" ]]; then
    if [[ ! -f "$SCALESIM_ENV_ACTIVATE" ]]; then
        echo "SCALE-Sim environment activation script not found: $SCALESIM_ENV_ACTIVATE" >&2
        exit 2
    fi
    # shellcheck source=/dev/null
    source "$SCALESIM_ENV_ACTIVATE"
fi

python_bin="${SCALESIM_PYTHON:-python3}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Python executable not found: $python_bin" >&2
    exit 2
fi

mode="${SCALESIM_MODE:-test}"
workloads="${SCALESIM_WORKLOADS:-hardware_sim/scalesim/workloads}"
layout="${SCALESIM_LAYOUT:-../scale-sim-v3/layouts/conv_nets/test.csv}"
config="${SCALESIM_CONFIG:-hardware_sim/scalesim/scale_user_bw.cfg}"
output="${SCALESIM_OUTPUT:-results/hardware_sim/scalesim}"
array_sizes="${SCALESIM_ARRAY_SIZES:-16,64,128}"
scenarios="${SCALESIM_SCENARIOS:-baseline}"
bandwidth="${SCALESIM_BANDWIDTH:-10}"
scalesim_root="${SCALESIM_ROOT:-}"

case "$mode" in
    test|sweep) ;;
    *)
        echo "SCALESIM_MODE must be 'test' or 'sweep'; got $mode" >&2
        exit 2
        ;;
esac

if [[ "$workloads" != /* ]]; then workloads="$repo_root/$workloads"; fi
if [[ "$layout" != /* ]]; then layout="$repo_root/$layout"; fi
if [[ "$config" != /* ]]; then config="$repo_root/$config"; fi
if [[ "$output" != /* ]]; then output="$repo_root/$output"; fi

if [[ ! -d "$workloads" ]]; then
    echo "SCALE-Sim workload directory not found: $workloads" >&2
    exit 2
fi
if [[ ! -f "$workloads/original_gemm.csv" ]]; then
    echo "Required ResNet workload is missing: $workloads/original_gemm.csv" >&2
    echo "Generate or transfer the real GEMM workload CSV before submitting this job." >&2
    exit 2
fi
if [[ ! -f "$layout" ]]; then
    echo "SCALE-Sim layout file not found: $layout" >&2
    exit 2
fi
if [[ ! -f "$workloads/lora_gemm.csv" && "$scenarios" != "baseline" ]]; then
    echo "LoRA workload is missing: $workloads/lora_gemm.csv" >&2
    exit 2
fi

if [[ -z "$scalesim_root" || ! -d "$scalesim_root/.git" ]]; then
    echo "Set SCALESIM_ROOT to the SCALE-Sim v3 source checkout pinned by this project." >&2
    exit 2
fi
expected_revision="$(tr -d '[:space:]' < "$repo_root/hardware_sim/scalesim/SCALESIM_V3_REVISION")"
actual_revision="$(git -C "$scalesim_root" rev-parse HEAD)"
if [[ "$actual_revision" != "$expected_revision" ]]; then
    echo "SCALE-Sim checkout revision mismatch: expected $expected_revision, got $actual_revision" >&2
    exit 2
fi

"$python_bin" - "$scalesim_root" <<'PY'
import pathlib
import sys
import scalesim

root = pathlib.Path(sys.argv[1]).resolve()
module_path = pathlib.Path(scalesim.__file__).resolve()
if root not in module_path.parents:
    raise SystemExit(f"Python imported SCALE-Sim from {module_path}, not pinned checkout {root}")
from scalesim.scale_config import scale_config

print(f"SCALE-Sim v3 source verified: {module_path}")
print(f"Config parser: {scale_config.__module__}")
PY

args=(
    --workloads "$workloads"
    --config "$config"
    --layout "$layout"
    --out "$output"
    --array-sizes "$array_sizes"
    --scenarios "$scenarios"
    --bandwidth "$bandwidth"
)
if [[ "$mode" == "test" ]]; then
    args+=(--test-run)
fi

echo "Starting SCALE-Sim mode=$mode job_id=${SLURM_JOB_ID:-local} node=${SLURMD_NODENAME:-local}"
"$python_bin" -m hardware_sim.scalesim.run_gemm "${args[@]}"
