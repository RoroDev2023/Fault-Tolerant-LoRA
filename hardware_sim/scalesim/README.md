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

### One-case test run

The runner supports a one-case mode so you can test the simulator integration
without running all array sizes and scenarios:

```bash
python hardware_sim/scalesim/run_gemm.py --test-run
```

By default this selects only the 16x16 baseline. To select a different single
case, provide the desired item first:

```bash
python hardware_sim/scalesim/run_gemm.py \
  --test-run \
  --array-sizes 64,16,128 \
  --scenarios clustered_lora,faulted_lora,baseline
```

This runs just the 64x64 clustered+LoRA case. `--test-run` uses the first array
size and first scenario from the respective options. Add `--dry-run` to verify
the selection without invoking SCALE-Sim:

```bash
python hardware_sim/scalesim/run_gemm.py --test-run --dry-run
```

### Full sweep

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

## Pipeline

1. Prepare GEMM workload CSVs in the workload directory. The baseline uses
   `original_gemm.csv`; the clustered+LoRA case also needs `lora_gemm.csv`;
   the faulted+LoRA case uses the same adapter workload.
2. The runner generates one SCALE-Sim config per requested square PE array
   size, using the selected fixed bandwidth.
3. For each requested case, it runs SCALE-Sim with the selected config,
   workload CSV, and layout, then reads each `COMPUTE_REPORT.csv`.
4. It sums the reported cycle counts and writes `sweep_summary.csv`; full
   simulator reports are kept under the corresponding `pe<size>/<scenario>/`
   output directory.

Workload generation is not yet automated by this package, and the repository
does not include the GEMM workload CSVs. A real `--test-run` therefore requires
at least `original_gemm.csv`, the external SCALE-Sim installation, and a valid
layout file. `--dry-run` checks only the requested case selection and does not
validate those runtime inputs.

## Important assumptions

- This is a hardware what-if layer, not a replacement for the model-level accuracy checks.
- The external SCALE-Sim v3 package still needs to be installed in a dedicated environment.
- Faulted scenarios use the same GEMM workload and therefore have the same simulated cycles as clustered+LoRA until explicit fault-cycle hardware is added.
- Baseline runs simulate the baseline workload once. LoRA scenarios simulate the base workload and then the adapter workload; these cycle counts are reported as a serial sum, not as end-to-end inference latency.
