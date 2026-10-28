# Next stage: compressed-weight representation accounting

The existing run measures 21 base + 42 LoRA GEMM operations. This helper is a
**storage-footprint accounting step only**; it does not change the SCALE-Sim
scheduler and does not claim additional or saved cycles.

From the `Fault-Tolerant-LoRA` repository root (after running
`extract_resnet_gemm.py`):

```sh
python hardware_sim/scalesim/analyze_cluster_storage.py
cat results/hardware_sim/scalesim/cluster_storage_summary.json
```

Defaults: one index per scalar weight, uint8 indices (8-bit), k=16 centroids per
Conv/Linear tensor, 4-byte FP32 dense weights and centroids, FP32 LoRA weights.
**Verify actual checkpoint/storage dtypes before paper reporting.** The software
may reconstruct a dense tensor before inference: this helper models a possible
compressed storage layout, *not necessarily current software DRAM behavior*.

For an **idealized** packed 4-bit index alternative (NOT the current uint8
representation), run:

```sh
python hardware_sim/scalesim/analyze_cluster_storage.py \
  --index-bits 4 --out results/hardware_sim/scalesim/packed4
```

Why this is useful: it quantifies the total bytes a hypothetical compression-
aware memory system could store and, in principle, transfer. It does not yet
model placement of the index decoder, centroid read latency, SRAM buffer
contention or timing, capacity/traffic reuse, DRAM bandwidth, or memory faults.
To model cycle counts next, first specify an architecture (on-demand lookups vs.
pre-decompress) and implement its requests in the pinned SCALE-Sim memory path.
For silent persistent faulted indices, keep the same timing unless an explicit
error detector/retry mechanism is included.
