# Step 3: fixed-bandwidth baseline and compressed DRAM traffic counterfactual

This package adds **two scripts** to `hardware_sim/scalesim` without changing any
model/training code or modifying SCALE-Sim. Run from the `Fault-Tolerant-LoRA`
repository root on `feat/resnet-scalesim` after previous steps completed.

## A. Fixed-bandwidth baseline (actual SCALE-Sim run)

The default config is `InterfaceBandwidth: CALC` and chooses sufficient
bandwidth to avoid stalls. Create a copy with `USER` mode:

```bash
python hardware_sim/scalesim/make_fixed_bw_config.py --bandwidth 10

../scale-sim-v3/.venv/bin/python \
  hardware_sim/scalesim/run_gemm.py \
  --config hardware_sim/scalesim/scale_32x32_user_bw.cfg \
  --layout ../scale-sim-v3/layouts/conv_nets/test.csv \
  --out results/hardware_sim/scalesim_fixed_bw

cat results/hardware_sim/scalesim_fixed_bw/comparison.csv
```

This uses SCALE-Sim's standard dense base/GEMM representation, not compressed
indices; `USER` mode tests sensitivity to the configured bandwidth. `Bandwidth`
in SCALE-Sim is measured in its **data elements per cycle**; don't interpret 10
as 10 bytes/cycle without calibrating element size.

## B. DRAM traffic from the existing CALC reports (postprocessing only)

The earlier `run_gemm.py` wrote its `DETAILED_ACCESS_REPORT.csv` reports
below `results/hardware_sim/scalesim/base/` and `.../adapters/`.

```bash
python hardware_sim/scalesim/estimate_dram_traffic.py
cat results/hardware_sim/scalesim/traffic_8bit/dram_traffic_summary.json

python hardware_sim/scalesim/estimate_dram_traffic.py \
  --index-bits 4 \
  --out results/hardware_sim/scalesim/traffic_4bit
```

This compares **DRAM filter data transferred**, under these assumptions:
- Each dense filter read corresponds to one 4-byte scalar weight;
- 8-bit indices in baseline compressed representation, 4-bit packed alternative;
- Same filter-request count and reuse schedule as standard SCALE-Sim;
- 16 FP32 centroids for each layer are loaded from DRAM *once* per layer and
  assumed resident thereafter in a reliable on-chip codebook buffer;
- Every adapter GEMM contributes its own DRAM IFMAP/FILTER/OFMAP access counts.

**DO NOT** interpret the output as actual DRAM bytes already transferred by an
index-aware simulator, actual codebook decode cycles, or recovered inference
latency. 4-bit packing requires additional coalescing/decoding hardware.

The next *research implementation* is to extend SCALE-Sim's filter buffer
`service_reads` / prefetch pipeline so index requests and codebook residency
are represented *before* it computes memory stalls. Persistent silent faults
should use the same access schedule unless explicit correction/retries exist.

## Sanity checks

- Existing reported total GEMM cycles in CALC mode should remain untouched.
- Same-dataflow fault-free and silently faulted scenarios should have identical
  cycle count when no detection/correction hardware is modeled.
- An index representation may reduce DRAM byte traffic without reducing the
  number of logical weight operands supplied to the PEs.
- 4-bit packing is a **hardware what-if**, not Rahim's current uint8 tensors.
