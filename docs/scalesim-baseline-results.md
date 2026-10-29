# SCALE-Sim Baseline ResNet Sweep Results

## Summary

The baseline ResNet-18 GEMM workload was simulated with SCALE-Sim v3 on three
square systolic arrays. All three runs completed successfully in Slurm job
`21294311` in 19 minutes 51 seconds.

| Array | Compute cycles | Prefetch-inclusive cycles | Compute-cycle reduction vs. 16x16 |
|---|---:|---:|---:|
| 16x16 | 5,554,092 | 5,709,585 | Reference |
| 64x64 | 3,926,309 | 4,085,929 | 29.3% |
| 128x128 | 3,717,791 | 3,883,996 | 33.1% |

The 128x128 array had the lowest cycle count. Increasing the array from 64x64
to 128x128 reduced compute cycles by a further 208,518 (5.3%), compared with
the 29.3% reduction from 16x16 to 64x64. This suggests diminishing cycle
reductions at the larger array size for this workload and these simulator
settings.

## Cycle comparison

Each `#` represents approximately 0.2 million cycles. Bars are scaled
independently within each metric.

```text
Compute cycles
16x16   ############################  5.554 M
64x64   ####################          3.926 M
128x128 ###################           3.718 M

Prefetch-inclusive cycles
16x16   ############################  5.710 M
64x64   ####################          4.086 M
128x128 ###################           3.884 M
```

## Run setup

- **Workload:** project ResNet-18 baseline topology, batch-1 inference at the
  configured 96x96 input resolution; 20 convolution GEMMs plus the classifier
  GEMM (21 layers total).
- **Scenarios:** baseline only. No LoRA adapter or injected-fault scenario was
  included.
- **Arrays:** 16x16, 64x64, and 128x128 processing elements.
- **SCALE-Sim:** pinned v3 source revision from
  [`hardware_sim/scalesim/SCALESIM_V3_REVISION`](../hardware_sim/scalesim/SCALESIM_V3_REVISION).
- **Configuration:** output-stationary dataflow, user bandwidth 10, and
  128 KiB each for ifmap, filter, and ofmap SRAM.
- **Environment:** dedicated SCALE-Sim Python environment with NumPy 1.26.4.
- **Execution:** CPU Slurm job `21294311`; completed with exit code 0 in
  00:19:51.

## Interpretation and limitations

These are simulated cycle counts for the modeled GEMM operations, not wall
clock inference latency or measurements from physical systolic-array
hardware. The topology excludes non-GEMM operations such as pooling,
activations, batch normalization, and residual additions. It also does not
model LoRA indexing/codebook decode, fault correction, or other system-level
overheads. The configured bandwidth and SRAM sizes are modeling assumptions;
they were not measured or confirmed as properties of NYU hardware.

The prefetch-inclusive cycle count includes the reported prefetch contribution
and should not be confused with elapsed job time. Slurm job duration was
19 minutes 51 seconds for the complete three-array sweep.

## Result files

The cluster output is under `results/hardware_sim/scalesim/`:

- `sweep_summary.csv` contains the totals shown above.
- Per-array `COMPUTE_REPORT.csv` files are under `pe16/`, `pe64/`, and
  `pe128/`, each within `baseline_resnet/base/`.

The runner writes these generated results to the ignored `results/` directory;
they are not included in Git unless copied or added separately.
