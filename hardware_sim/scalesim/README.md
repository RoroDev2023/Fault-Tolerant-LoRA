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

The config generator emits the actual SCALE-Sim v3 INI schema, and this
integration pins the upstream source commit in `SCALESIM_V3_REVISION`.
Generated per-array configs are rewritten on each run so old configs from
earlier incompatible templates cannot be silently reused.

The script is intentionally conservative: it reports GEMM-only schedules, which matches the current model of the project. It does not attempt to model index lookup, codebook decoding, or fault-correction latency unless those components are added to the simulator backend.

## Usage

### One-case test run

The runner supports a one-case mode so you can test the simulator integration
without running all array sizes and scenarios:

```bash
python -m hardware_sim.scalesim.run_gemm --test-run
```

By default this selects only the 16x16 baseline. To select a different single
case, provide the desired item first:

```bash
python -m hardware_sim.scalesim.run_gemm \
  --test-run \
  --array-sizes 64,16,128 \
  --scenarios clustered_lora,faulted_lora,baseline
```

For the config format validated by this repository, use the pinned SCALE-Sim v3
checkout specified by `SCALESIM_V3_REVISION`. The generated config sections
and keys follow that revision's `scale_config.read_conf_file` parser. The
runner loads each generated config through the installed SCALE-Sim parser and
checks the array dimensions, output-stationary dataflow, and requested user
bandwidth before launching a simulation. Use a dedicated SCALE-Sim environment
with NumPy 1.26.4: the pinned v3 source fails in its prefetch path with NumPy
2.x, while the 16x16 ResNet smoke run completes with NumPy 1.26.4.

This runs just the 64x64 clustered+LoRA case. `--test-run` uses the first array
size and first scenario from the respective options. Add `--dry-run` to verify
the selection without invoking SCALE-Sim:

```bash
python hardware_sim/scalesim/run_gemm.py --test-run --dry-run
```

### Full sweep

```bash
python -m hardware_sim.scalesim.run_gemm \
  --workloads hardware_sim/scalesim/workloads \
  --config hardware_sim/scalesim/scale_user_bw.cfg \
  --layout ../scale-sim-v3/layouts/conv_nets/test.csv \
  --out results/hardware_sim/scalesim \
  --array-sizes 16,64,128 \
  --scenarios baseline,clustered_lora,faulted_lora
```

For a dry-run with no simulator execution:

```bash
python -m hardware_sim.scalesim.run_gemm --dry-run
```

## Pipeline

1. `extract_resnet_gemm.py` traces the project's torchvision ResNet-18 with
   the configured input size and writes `original_gemm.csv`. Each Conv2d maps
   to `M=output height*output width*batch`, `N=output channels`, and
   `K=input channels*kernel height*kernel width`; the Linear layer maps to its
   flattened `M,N,K` dimensions. The checked-in baseline config uses 96x96
   inputs and the default topology represents batch size 1.
2. The CSV uses SCALE-Sim v3's GEMM topology format: layer name, M, N, K,
   dense sparsity ratio `1:1`, and the trailing empty column expected by the
   upstream parser.
3. The runner generates one SCALE-Sim config per requested square PE array
   size, using the selected fixed bandwidth.
4. For each requested case, it runs SCALE-Sim with the selected config,
   workload CSV, and layout, then reads each `COMPUTE_REPORT.csv`.
5. It sums the reported cycle counts and writes `sweep_summary.csv`; full
   simulator reports are kept under the corresponding `pe<size>/<scenario>/`
   output directory.

Regenerate the baseline workload after changing the input resolution or batch
size:

```bash
python -m hardware_sim.scalesim.extract_resnet_gemm \
  --config configs/resnet18_cifar10.json \
  --batch-size 1
```

An existing clean checkpoint can be used to take the exact saved preprocessing
resolution:

```bash
python -m hardware_sim.scalesim.extract_resnet_gemm \
  --checkpoint checkpoints/resnet18-cifar10-clean-seed42/best.pt
```

This topology models Conv2d and Linear MACs only. Batch normalization,
activations, pooling, residual additions, data movement outside SCALE-Sim's
modeled operands, and non-GEMM operations are excluded. It is a single-image
inference topology by default, not the training batch's dataflow or latency.
The clustered/faulted model weights do not change these GEMM shapes.

## Important assumptions

- This is a hardware what-if layer, not a replacement for the model-level accuracy checks.
- The external SCALE-Sim v3 package still needs to be installed in a dedicated environment.
- Faulted scenarios use the same GEMM workload and therefore have the same simulated cycles as clustered+LoRA until explicit fault-cycle hardware is added.
- Baseline runs simulate the baseline workload once. LoRA scenarios simulate the base workload and then the adapter workload; these cycle counts are reported as a serial sum, not as end-to-end inference latency.
