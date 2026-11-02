# SCALE-Sim Clustered ResNet + LoRA Sweep Results

## Summary

The baseline ResNet-18 GEMM workload and the clustered ResNet + LoRA adapter
workload were simulated with SCALE-Sim v3 on four square systolic arrays
(16x16, 32x32, 64x64, 128x128). All eight runs completed in Slurm job
`21623269` (exit code 0, elapsed 01:24:47).

- The LoRA adapters add **6.4% to 11.4%** compute cycles over the dense
  baseline. The overhead grows with array size.
- The adapters cost a roughly constant **0.35 to 0.42 M cycles** at every array
  size. Larger arrays speed up the dense layers but not the adapter GEMMs, so
  the adapters' relative share grows.
- The adapter GEMMs (rank 4) map poorly onto the arrays: their estimated compute
  utilization falls from about 13.9% on 16x16 to about 0.7% on 128x128.
- The baseline cycle counts at 16x16, 64x64, and 128x128 match the earlier
  baseline sweep (job `21294311`) exactly.

## Raw sweep summary

Exact output of `column -s, -t sweep_summary.csv` from the cluster:

```text
array_size  scenario        description                          status    base_workload      lora_workload      total_cycles  prefetch_cycles  mean_mapping_efficiency_percent  mean_compute_utilization_percent  mean_inactive_pe_equivalents  overhead_vs_baseline  config
16          baseline        Dense ResNet baseline                executed  original_gemm.csv  original_gemm.csv  5554092       5709585          79.05505952380952                73.90277752076484                 66.80888954684202             0.0                   /home/ms16082/Fault-Tolerant-LoRA/hardware_sim/scalesim/scale_16x16_user_bw.cfg
16          clustered_lora  Clustered ResNet with LoRA adapters  executed  original_gemm.csv  lora_gemm.csv      5908904       6215831          59.30059523809524                33.9018081818275                  169.2113710545216             6.3883                /home/ms16082/Fault-Tolerant-LoRA/hardware_sim/scalesim/scale_16x16_user_bw.cfg
32          baseline        Dense ResNet baseline                executed  original_gemm.csv  original_gemm.csv  5734637       5891051          65.37388392857143                57.35228181926059                 436.7126341707716             0.0                   /home/ms16082/Fault-Tolerant-LoRA/hardware_sim/scalesim/scale_32x32_user_bw.cfg
32          clustered_lora  Clustered ResNet with LoRA adapters  executed  original_gemm.csv  lora_gemm.csv      6107065       6424185          46.310763888888886               22.83159253997395                 790.2044923906667             6.4944                /home/ms16082/Fault-Tolerant-LoRA/hardware_sim/scalesim/scale_32x32_user_bw.cfg
64          baseline        Dense ResNet baseline                executed  original_gemm.csv  original_gemm.csv  3926309       4085929          58.419363839285715               46.41611740900401                 2194.7958309271958            0.0                   /home/ms16082/Fault-Tolerant-LoRA/hardware_sim/scalesim/scale_64x64_user_bw.cfg
64          clustered_lora  Clustered ResNet with LoRA adapters  executed  original_gemm.csv  lora_gemm.csv      4280653       4603719          40.164620535714285               17.03926414176457                 3398.0717407533234            9.0249                /home/ms16082/Fault-Tolerant-LoRA/hardware_sim/scalesim/scale_64x64_user_bw.cfg
128         baseline        Dense ResNet baseline                executed  original_gemm.csv  original_gemm.csv  3717791       3883996          32.718680245535715               22.391734580146785                12715.338206388751            0.0                   /home/ms16082/Fault-Tolerant-LoRA/hardware_sim/scalesim/scale_128x128_user_bw.cfg
128         clustered_lora  Clustered ResNet with LoRA adapters  executed  original_gemm.csv  lora_gemm.csv      4139827       4475878          22.267717633928573               7.937540324802861                 15083.5133931843              11.3518               /home/ms16082/Fault-Tolerant-LoRA/hardware_sim/scalesim/scale_128x128_user_bw.cfg
```

## Cycle overhead from the LoRA adapters

| Array | Baseline compute cycles | Clustered+LoRA compute cycles | Adapter cycles | Overhead | Baseline prefetch-incl. | Clustered+LoRA prefetch-incl. | Prefetch-incl. overhead |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16x16 | 5,554,092 | 5,908,904 | 354,812 | 6.39% | 5,709,585 | 6,215,831 | 8.87% |
| 32x32 | 5,734,637 | 6,107,065 | 372,428 | 6.49% | 5,891,051 | 6,424,185 | 9.05% |
| 64x64 | 3,926,309 | 4,280,653 | 354,344 | 9.02% | 4,085,929 | 4,603,719 | 12.67% |
| 128x128 | 3,717,791 | 4,139,827 | 422,036 | 11.35% | 3,883,996 | 4,475,878 | 15.24% |

"Adapter cycles" is clustered+LoRA minus baseline compute cycles at the same
array size. The adapters add 3.8% of the base multiply-accumulates
(12.6 M vs 0.33 G) but 6–11% of the cycles. The overhead is higher still
once prefetch is included, because each skinny adapter GEMM pays its own
fill/drain and data-movement cost.

## Array utilization

The summary means are unweighted averages over layers: 21 base layers for
baseline, and 21 base + 42 adapter layers for clustered+LoRA. The adapter-only
column is derived as (3 × clustered − baseline) / 2. It is an inferred value
and can be checked against `layer_metrics.csv`.

| Array | Base mapping eff. | Base compute util. | Adapter mapping eff. (derived) | Adapter compute util. (derived) | Adapter inactive PEs (of total) |
|---|---:|---:|---:|---:|---:|
| 16x16 | 79.1% | 73.9% | 49.4% | 13.9% | 220 of 256 |
| 32x32 | 65.4% | 57.4% | 36.8% | 5.6% | 967 of 1,024 |
| 64x64 | 58.4% | 46.4% | 31.0% | 2.4% | 4,000 of 4,096 |
| 128x128 | 32.7% | 22.4% | 17.0% | 0.7% | 16,268 of 16,384 |

The rank-4 adapter GEMMs have N = 4 (`lora_A`) or K = 4 (`lora_B`), so they
occupy only a sliver of an output-stationary array. Larger arrays leave almost
every PE idle during adapter layers. This explains why the adapter cycle cost
stays flat instead of shrinking as the array grows.

## Baseline scaling across array sizes

| Array | Baseline compute cycles | Change vs. 16x16 |
|---|---:|---:|
| 16x16 | 5,554,092 | Reference |
| 32x32 | 5,734,637 | +3.3% |
| 64x64 | 3,926,309 | −29.3% |
| 128x128 | 3,717,791 | −33.1% |

32x32 is slightly *slower* than 16x16 on the dense baseline, despite 4× the
PEs. Its mean compute utilization drops from 73.9% to 57.4%. This run does not
show the cause. Fold boundaries for these layer shapes, or the fixed user
bandwidth of 10, are plausible explanations. The per-layer `COMPUTE_REPORT.csv`
files under `pe16/` and `pe32/` should show which layers regress.

## Run setup

- **Workloads:** `original_gemm.csv` (ResNet-18, 21 GEMM layers, batch 1,
  96x96 input) and `lora_gemm.csv` (rank-4 LoRA adapters on 21 targets,
  42 GEMMs: one `lora_A` and one `lora_B` per target).
- **Scenarios:** `baseline` (dense ResNet) and `clustered_lora` (base workload
  followed by the adapter workload).
- **Arrays:** 16x16, 32x32, 64x64, 128x128 processing elements.
- **SCALE-Sim:** pinned v3 source revision from
  [`hardware_sim/scalesim/SCALESIM_V3_REVISION`](../hardware_sim/scalesim/SCALESIM_V3_REVISION);
  layout `layouts/conv_nets/test.csv`.
- **Configuration:** generated `scale_<N>x<N>_user_bw.cfg` files, user
  bandwidth 10.
- **Execution:** CPU Slurm job `21623269` (1 CPU, 8 GB); completed with exit
  code 0 in 01:24:47.

## Interpretation and limitations

- `clustered_lora` cycles are a serial sum of the base and adapter workloads,
  not end-to-end inference latency. A design that overlaps or fuses the adapter
  path with the base layer would see a different cost.
- The cycle counts do not depend on clustering or fault patterns. The base
  workload is the same dense ResNet in both scenarios, and fault/codebook
  overheads are not modeled. The overhead shown here is the cost of the
  adapter GEMMs alone.
- Utilization means are unweighted per-layer averages, so short layers count
  as much as long ones. Use `layer_metrics.csv` for cycle-weighted views.
- As in the baseline sweep, non-GEMM operations (pooling, activations, batch
  norm, residual adds) are excluded. Bandwidth and SRAM sizes are modeling
  assumptions, not NYU hardware properties.

## Result files

The cluster output is under
`/home/ms16082/Fault-Tolerant-LoRA/results/hardware_sim/scalesim-clustered-lora/`:

- `sweep_summary.csv` contains the totals shown above.
- `layer_metrics.csv` contains mapping efficiency, compute utilization, and
  inactive PEs per layer, tagged `base` or `lora`.
- `pe<size>/<scenario>/base/` and `.../lora/` contain the raw SCALE-Sim
  `COMPUTE_REPORT.csv` for each run.

These generated results live in the ignored `results/` directory and are not
in Git.
