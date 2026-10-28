# SCALE-Sim hardware sweep

This package adds a hardware-analysis layer for the Fault-Tolerant-LoRA project without changing the model-training workflow.

## What it does

The runner in `run_gemm.py` sweeps a small matrix of systolic-array sizes:

- 16x16
- 64x64
- 128x128

and evaluates three scenarios:

- baseline ResNet
- clustered ResNet + LoRA
- clustered ResNet + LoRA + injected faults

The script is intentionally conservative: it reports GEMM-only schedules, which matches the current model of the project. It does not attempt to model index lookup, codebook decoding, or fault-correction latency unless those components are added to the simulator backend.

## Usage

```bash
python hardware_sim/scalesim/run_gemm.py \
  --workloads hardware_sim/scalesim/workloads \
  --config hardware_sim/scalesim/scale_user_bw.cfg \
  --layout ../scale-sim-v3/layouts/conv_nets/test.csv \
  --out results/hardware_sim/scalesim \
  --array-sizes 16,64,128 \
  --scenarios baseline,clustered_lora,faulted_lora
```

For a dry-run with no simulator execution:

```bash
python hardware_sim/scalesim/run_gemm.py --dry-run
```

## Important assumptions

- This is a hardware what-if layer, not a replacement for the model-level accuracy checks.
- The external SCALE-Sim v3 package still needs to be installed in a dedicated environment.
- Faulted scenarios are treated as the same GEMM schedule as the clustered+LoRA case until explicit fault-cycle hardware is added.
