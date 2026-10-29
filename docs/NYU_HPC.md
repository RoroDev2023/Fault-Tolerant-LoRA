# Running Fault-Tolerant-LoRA on NYU HPC

## What stays separate

Use your normal NYU SSH/login process to access the cluster, and submit jobs
manually with Slurm. The repository contains batch scripts that run existing
Python entry points; it does not store SSH credentials, log in remotely, or
submit jobs from Python. This keeps authentication and account-specific
resource choices under your control.

The scripts are templates, not a preconfigured NYU allocation. Confirm the
login host, account, partition, GPU request syntax/type, memory/time limits,
module names, and storage paths in the documentation for your NYU cluster.
Do not commit passwords, tokens, private keys, or account-specific secrets.

## 1. Put the repository and data on persistent shared storage

Clone or synchronize the project onto a filesystem visible to compute nodes.
Use a persistent project/scratch location for `checkpoints/` and `results/`;
do not rely on node-local temporary storage for experiment artifacts.

```bash
ssh YOUR_NYU_LOGIN@YOUR_NYU_LOGIN_HOST
cd /YOUR/PERSISTENT/PROJECT/PATH
git clone YOUR_REPOSITORY_URL Fault-Tolerant-LoRA
cd Fault-Tolerant-LoRA
```

If the repository is already checked out, update it to the commit you intend to
run. Transfer any existing datasets, checkpoints, audit artifacts, and sweep
patterns to the corresponding project-relative paths. `checkpoints/`, `data/`,
and `results/` are intentionally ignored by Git and are not included in a clone.

For a clean-baseline run, the runner can download CIFAR-10 and pretrained
weights with `FAULT_LORA_DOWNLOAD=1`; only do this if compute nodes have
network access and your cluster policy permits it. Otherwise stage the data
and pretrained checkpoint/cache before submitting.

## 2. Prepare a Linux/CUDA Python environment

The checked-in `requirements-macos-arm64.lock.txt` is for macOS and must not be
used on Linux. Load the Python/CUDA modules and install a PyTorch build
compatible with the cluster's driver and supported CUDA version using NYU's
documented installation instructions. Then install/verify the other project
dependencies in that same environment. The current repository does not yet
provide a validated Linux/CUDA lockfile.

Set the activation script and Python path for subsequent submissions. For
example, if using a virtual environment:

```bash
export FAULT_LORA_ENV_ACTIVATE=/path/to/venv/bin/activate
export FAULT_LORA_PYTHON=/path/to/venv/bin/python
```

The model job checks the requested device with `scripts/check_environment.py`
before launching a GPU workflow. `FAULT_LORA_DEVICE` defaults to `cuda`; set it
to `cpu` only if you intentionally want a CPU run.

## 3. Submit model stages

Run submissions from the repository root, and provide NYU-specific account,
partition, resource, and log options to `sbatch`. The examples below use
placeholders: replace them with values and resource syntax supported by your
NYU allocation.

```bash
mkdir -p "$HOME/slurm-logs"
sbatch \
  --account=YOUR_ACCOUNT \
  --partition=YOUR_GPU_PARTITION \
  --gres=gpu:1 \
  --time=HH:MM:SS \
  --mem=MEMORY \
  --output="$HOME/slurm-logs/%x-%j.out" \
  hpc/slurm/model_job.sbatch baseline configs/resnet18_cifar10.json
```

The model job accepts the following operation names and defaults:

| Operation | Default config | Purpose |
| --- | --- | --- |
| `baseline` | `configs/resnet18_cifar10.json` | Clean CIFAR-10 ResNet training |
| `clustered` | `configs/resnet18_cifar10_clustered.json` | Cluster the saved clean baseline |
| `faults` | `configs/resnet18_cifar10_faults.json` | Generate a static fault pattern |
| `fault-checks` | `configs/resnet18_cifar10_fault_checks.json` | Audit the fault artifacts |
| `fault-sweep` | `configs/resnet18_cifar10_fault_sweep.json` | Evaluate the fault-rate sweep |
| `lora` | `configs/resnet18_cifar10_lora.json` | Train/evaluate LoRA recovery |

You can pass a different config as the second positional argument. The batch
script creates a temporary config overlay, changing `device` from the checked-
in macOS value to `cuda` and optionally replacing `data_root`; it does not
modify the checked-in JSON. Set `FAULT_LORA_DATA_ROOT` before submission if the
dataset is stored outside the repository's `data/` directory:

```bash
export FAULT_LORA_DATA_ROOT=/YOUR/SHARED/DATA/PATH
```

You can test the operation/config selection locally without importing CUDA or
launching the Python experiment:

```bash
FAULT_LORA_DRY_RUN=1 bash hpc/slurm/model_job.sbatch baseline
```

By default the code writes checkpoints and results under the shared checkout's
`checkpoints/` and `results/` directories. This keeps existing relative
checkpoint/result references valid between workflow stages.

Submit dependent stages only after the preceding job succeeds:

1. `baseline`
2. `clustered`
3. `faults`
4. `fault-checks`
5. `fault-sweep`
6. `lora`

Alternatively, after confirming the cluster's Slurm dependency policy, submit
later jobs with `--dependency=afterok:JOB_ID`. The Python programs validate
source artifacts and will fail if an upstream stage is missing or incomplete.

To opt into network downloads for the clean baseline only, export
`FAULT_LORA_DOWNLOAD=1` before submitting. The default is no download.

## 4. Submit SCALE-Sim

SCALE-Sim runs as a separate CPU job and needs its own environment containing
the exact upstream SCALE-Sim v3 source revision recorded in
`hardware_sim/scalesim/SCALESIM_V3_REVISION`. The config generator uses v3's
`[general]`, `[architecture_presets]`, `[layout]`, `[sparsity]`, and
`[run_presets]` schema. Clone that exact revision and install it editable into
the SCALE-Sim environment:

```bash
git clone https://github.com/scalesim-project/scale-sim-v3.git /YOUR/PATH/scale-sim-v3
git -C /YOUR/PATH/scale-sim-v3 checkout "$(cat /PATH/TO/Fault-Tolerant-LoRA/hardware_sim/scalesim/SCALESIM_V3_REVISION)"
python -m pip install "numpy==1.26.4"
python -m pip install -e /YOUR/PATH/scale-sim-v3
export SCALESIM_ENV_ACTIVATE=/path/to/scalesim-venv/bin/activate
export SCALESIM_PYTHON=/path/to/scalesim-venv/bin/python
export SCALESIM_ROOT=/YOUR/PATH/scale-sim-v3
```

Keep this as a dedicated environment. The pinned SCALE-Sim v3 revision fails
during simulation with NumPy 2.x; its 16x16 ResNet smoke run was verified with
NumPy 1.26.4.

The batch script checks the checkout's Git revision and confirms Python imports
SCALE-Sim from that checkout before starting.

The baseline GEMM workload is generated from the actual torchvision ResNet-18
architecture at the 96x96 input resolution in the checked-in CIFAR-10 config.
The generated `hardware_sim/scalesim/workloads/original_gemm.csv` represents
batch-1 inference and is tracked in this repository. If you change input
resolution, regenerate it with:

```bash
python -m hardware_sim.scalesim.extract_resnet_gemm \
  --config configs/resnet18_cifar10.json \
  --batch-size 1
```

For clustered+LoRA scenarios, generate the adapter topology from the checked-in
ResNet and LoRA configs on a machine with the project PyTorch environment:

```bash
python -m hardware_sim.scalesim.extract_lora_gemm
```

This writes `hardware_sim/scalesim/workloads/lora_gemm.csv`, containing the A
and B adapter GEMMs for each configured Conv2d/Linear target. Transfer this
file along with the project files to Torch. The runner combines those adapter
cycles serially with the baseline ResNet GEMMs; it does not model clustering
compression or adapter numerical values. Set `SCALESIM_LAYOUT` to a valid
layout from your SCALE-Sim checkout. The job validates its input files and
simulator import before starting.

By default, the script runs a single test case (16x16 baseline), not the full
matrix:

```bash
export SCALESIM_LAYOUT=/PATH/TO/scale-sim-v3/layouts/conv_nets/test.csv
sbatch \
  --account=YOUR_ACCOUNT \
  --partition=YOUR_CPU_PARTITION \
  --time=HH:MM:SS \
  --mem=MEMORY \
  --cpus-per-task=4 \
  --output="$HOME/slurm-logs/%x-%j.out" \
  hpc/slurm/scalesim_job.sbatch
```

After confirming the one-case result, request the full sweep:

```bash
export SCALESIM_MODE=sweep
export SCALESIM_ARRAY_SIZES=16,64,128
export SCALESIM_SCENARIOS=clustered_lora
sbatch ... hpc/slurm/scalesim_job.sbatch
```

Optional inputs include `SCALESIM_WORKLOADS`, `SCALESIM_CONFIG`,
`SCALESIM_OUTPUT`, and `SCALESIM_BANDWIDTH`. `SCALESIM_CONFIG` determines
where per-array configs are written by the current runner. The output remains
a GEMM-only cycle estimate: index/codebook decode, fault correction, and
end-to-end model latency are not simulated.

## 5. Monitor jobs and preserve results

Use Slurm directly:

```bash
squeue -u YOUR_NYU_USER
scontrol show job JOB_ID
scancel JOB_ID
```

Read the log path passed through `sbatch --output`. Successful model runs also
record `config.hpc.json` and `hpc_job.json` beside their results, including the
effective config, Git revision, and Slurm job/allocation metadata. Copy
checkpoints and results from the persistent project filesystem before any
cluster cleanup policy removes them.

No commands in this guide have been submitted automatically. Validate paths,
environment modules, and resource requests for your account before submitting
real jobs.
