#!/usr/bin/env bash
# Submit from the repository root. Supply account, partition, time, memory,
# GPU, and log-output options to sbatch for the NYU cluster you use.
#SBATCH --job-name=fault-lora-model
#SBATCH --ntasks=1
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

repo_root="${FAULT_LORA_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
repo_root="$(cd "$repo_root" && pwd)"
cd "$repo_root"

if [[ -n "${FAULT_LORA_ENV_ACTIVATE:-}" ]]; then
    if [[ ! -f "$FAULT_LORA_ENV_ACTIVATE" ]]; then
        echo "Environment activation script not found: $FAULT_LORA_ENV_ACTIVATE" >&2
        exit 2
    fi
    # shellcheck source=/dev/null
    source "$FAULT_LORA_ENV_ACTIVATE"
fi

python_bin="${FAULT_LORA_PYTHON:-python3}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Python executable not found: $python_bin" >&2
    exit 2
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: sbatch [Slurm options] $0 OPERATION [CONFIG]" >&2
    echo "Operations: baseline, clustered, faults, fault-checks, fault-sweep, lora" >&2
    exit 2
fi

operation="$1"
config_path="${2:-}"
needs_cuda=0
case "$operation" in
    baseline)
        runner="scripts/run_clean_baseline.py"
        config_path="${config_path:-configs/resnet18_cifar10.json}"
        needs_cuda=1
        ;;
    clustered)
        runner="scripts/run_clustered_baseline.py"
        config_path="${config_path:-configs/resnet18_cifar10_clustered.json}"
        needs_cuda=1
        ;;
    faults)
        runner="scripts/generate_fault_pattern.py"
        config_path="${config_path:-configs/resnet18_cifar10_faults.json}"
        ;;
    fault-checks)
        runner="scripts/check_fault_injection.py"
        config_path="${config_path:-configs/resnet18_cifar10_fault_checks.json}"
        ;;
    fault-sweep)
        runner="scripts/run_fault_sweep.py"
        config_path="${config_path:-configs/resnet18_cifar10_fault_sweep.json}"
        needs_cuda=1
        ;;
    lora)
        runner="scripts/run_lora_recovery.py"
        config_path="${config_path:-configs/resnet18_cifar10_lora.json}"
        needs_cuda=1
        ;;
    *)
        echo "Unknown operation: $operation" >&2
        exit 2
        ;;
esac

if [[ "$config_path" != /* ]]; then
    config_path="$repo_root/$config_path"
fi
if [[ ! -f "$config_path" ]]; then
    echo "Config not found: $config_path" >&2
    exit 2
fi

tmp_root="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
mkdir -p "$tmp_root"
job_tmp="$(mktemp -d "$tmp_root/fault-lora-${SLURM_JOB_ID:-local}.XXXXXX")"
trap 'rm -rf "$job_tmp"' EXIT
effective_config="$job_tmp/config.json"
config_args=(--input "$config_path" --output "$effective_config")

if [[ "$needs_cuda" -eq 1 ]]; then
    device="${FAULT_LORA_DEVICE:-cuda}"
    if [[ "$device" != "cuda" && "$device" != "cpu" ]]; then
        echo "FAULT_LORA_DEVICE must be cuda or cpu on this Linux job; got $device" >&2
        exit 2
    fi
    config_args+=(--device "$device")
    if [[ -n "${FAULT_LORA_DATA_ROOT:-}" ]]; then
        config_args+=(--data-root "$FAULT_LORA_DATA_ROOT")
    fi

    if [[ "${FAULT_LORA_DRY_RUN:-0}" != "1" ]]; then
        "$python_bin" scripts/check_environment.py --device "$device"
    fi
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
    export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
fi

"$python_bin" scripts/prepare_hpc_config.py "${config_args[@]}"

run_args=(--config "$effective_config")
if [[ "$operation" == "baseline" && "${FAULT_LORA_DOWNLOAD:-0}" == "1" ]]; then
    run_args+=(--download)
fi

if [[ "${FAULT_LORA_DRY_RUN:-0}" == "1" ]]; then
    printf 'Dry run: would execute %q' "$python_bin"
    printf ' %q' "$runner" "${run_args[@]}"
    printf '\n'
    exit 0
fi

echo "Starting operation=$operation job_id=${SLURM_JOB_ID:-local} node=${SLURMD_NODENAME:-local}"
"$python_bin" "$runner" "${run_args[@]}"
"$python_bin" scripts/record_slurm_metadata.py --repo-root "$repo_root" --config "$effective_config"
