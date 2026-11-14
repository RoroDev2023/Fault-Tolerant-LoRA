# SCALE-Sim Memory-System Evaluation for ResNet-18

## Scope and reporting conventions

This study evaluates the original ResNet-18 workload using two distinct SCALE-Sim v3 memory models: an **upstream reference** and an **experimental memory model**. The workload consists of **21 Conv/Linear GEMM-equivalent layers**, executed and summed serially for one input of shape `(1, 3, 96, 96)`.

The two result roots are independent. Results are ordered hierarchically by **dataflow (WS → IS → OS), PE-array size, SRAM capacity per operand, and bandwidth**; bandwidth varies first within an otherwise identical configuration. Unlisted combinations have no verified result in this study.

**Cycle definition:** `Total cycles` denotes the sum of per-layer simulated compute cycles **including memory stalls**, but excluding the separately reported initial-prefetch contribution. The number is *not* an end-to-end physical inference latency.

### Result directory layout

```text
hardware_sim/scalesim/
├── README.md
├── patches/
│   ├── upstream-source.txt
│   ├── upstream-compat.patch
│   ├── experimental-source.txt
│   ├── experimental-memory-model.patch
│   └── experimental-commits.txt
└── experiments/
    ├── upstream_reference/
    │   ├── configs/
    │   └── results.csv
    ├── experimental_memory_model/
    │   ├── runtime_settings.json
    │   ├── results.csv
    │   ├── runs/
    │   └── offline_first_use/  # Single-layer study, to be archived
    └── workloads/
        └── original_gemm.csv   # Shared input, not a result
```

The directory layout above is the **proposed organization** for this branch. At the time of drafting, the committed results are under `experiments/upstream/` and `experiments/modified/`; those directories should be renamed only after the README and result inventory are approved.

---

## 1. Upstream Reference — `experiments/upstream_reference/`

### Memory servicing in the upstream implementation

This reference uses upstream SCALE-Sim. Two NumPy compatibility edits are recorded in `patches/upstream-compat.patch`; they are not proposed memory-system architectural changes.

1. **Demand and prefetch generation.** The selected WS/IS/OS dataflow produces PE demand traces and operand-specific prefetch-address sequences for the model's input and filter buffers.
2. **Active/prefetch buffers.** The read scratchpad is modeled with an active portion and a prefetch portion. A prefetch fills the next window of operand addresses before that window is activated.
3. **Address membership.** The upstream `read_buffer` groups addresses from its prefetch sequence into sets (`hashed_buffer`) and checks membership over the currently active set-window. This is **not** a tree-PLRU or other CPU cache-way replacement policy.
4. **Miss and refill handling.** On an active-buffer miss, the simulator advances through successive prefetch windows and transfers data from the modeled backing memory until the requested address is present. Its ordered cursor can wrap around the operand sequence, causing repeated backing-memory transfers.
5. **Timing and stalls.** Modeled backing-memory bandwidth and completion times determine when data becomes available; the memory-service path charges the corresponding waits as stalls. PR #169 corrects errors in how some of these waits were counted.

The resulting traffic and stalls are properties of this *simulation algorithm*; repeated full-operand refetches should not be assumed to be unavoidable in physical hardware.

### Full-network results

**Common conditions:** ResNet-18 base GEMMs (21 layers), USER bandwidth mode, upstream **one-byte-per-element SRAM-capacity convention**.

| Dataflow | PE array | SRAM each (KiB) | BW (words/cycle) | Total cycles | Stall cycles | Nonstall cycles | Stall share |
|:--|:--|--:|--:|--:|--:|--:|--:|
| WS | 64×64 | 64 | 160 | 5,917,558 | 5,314,977 | 602,581 | 89.82% |
| WS | 64×64 | 128 | 160 | 3,426,191 | 2,823,610 | 602,581 | 82.41% |
| WS | 128×128 | 128 | 160 | 3,113,170 | 2,816,073 | 297,097 | 90.46% |

**Observations:** At 64×64 PEs and BW160, doubling each SRAM from 64 to 128 KiB reduces total cycles by **42.1%**, with unchanged nonstall cycles. At 128 KiB and BW160, increasing the array from 64×64 to 128×128 reduces nonstall cycles by **50.7%**, but stalls remain approximately **2.82 million**, limiting total-cycle improvement to **9.1%**.

**Artifacts:** `upstream_reference/configs/*.cfg`, `upstream_reference/results.csv`, `patches/upstream-source.txt`, and `patches/upstream-compat.patch`.

---

## 2. Experimental Memory Model — `experiments/experimental_memory_model/`

### Changes relative to the upstream implementation

The experimental source is independently pinned using `patches/experimental-source.txt` and archived through `patches/experimental-memory-model.patch`. The changes affect memory simulation as follows:

1. **Explicit prefetch cursor:** tracks progression through the flattened prefetch sequence, including row/column offsets and wraparound. This affects which addresses are transferred during refills.
2. **Exact active-window residency (optional):** tracks the addresses actually present in the active window using explicit insertion/eviction order and occupancy. Its behavior is **FIFO-like**.
3. **FP32 SRAM-capacity accounting:** `SCALESIM_WORD_SIZE_BYTES=4` uses four bytes per logical element when computing capacity. With half of a 64 KiB scratchpad serving as the active portion, that active portion accommodates **8,192 FP32 elements**.
4. **Optional offline first-use prefetch:** `SCALESIM_WS_IFMAP_PREFETCH=first_use` reorders **WS IFMAP** prefetches using the **complete PE demand trace**, so addresses are fetched in first-use order. This is an *offline, future-informed scheduling experiment*, **not** a validated online hardware prefetcher.
5. **Trace and compatibility handling:** separate changes improve trace accumulation and consumption and address output-cycle accounting. These should be assessed separately from changes to modeled memory behavior.

### Full-network results — modified, legacy prefetch ordering

**Common conditions:** ResNet-18 base GEMMs (21 layers), 64×64 PE array, USER bandwidth mode; intended FP32 capacity accounting, exact active-window tracking, and **legacy** prefetch order. **None of these seven full-network runs used offline first-use prefetch.**

| Dataflow | PE array | SRAM each (KiB) | BW (words/cycle) | Total cycles | Stall cycles | Nonstall cycles | Stall share |
|:--|:--|--:|--:|--:|--:|--:|--:|
| WS | 64×64 | 64 | 10 | 6,190,294 | 5,587,713 | 602,581 | 90.27% |
| WS | 64×64 | 64 | 160 | 654,382 | 51,801 | 602,581 | 7.92% |
| WS | 64×64 | 512 | 40 | 621,704 | 19,123 | 602,581 | 3.08% |
| IS | 64×64 | 64 | 10 | 68,093,013 | 67,683,850 | 409,163 | 99.40% |
| IS | 64×64 | 512 | 40 | 7,647,476 | 7,238,313 | 409,163 | 94.65% |
| OS | 64×64 | 64 | 10 | 9,266,499 | 9,031,026 | 235,473 | 97.46% |
| OS | 64×64 | 512 | 40 | 936,370 | 700,897 | 235,473 | 74.85% |

**Observations:** In this model, simultaneous increases from 64 to 512 KiB SRAM and BW10 to BW40 reduce WS total cycles from **6.19M to 0.62M**. Because capacity and bandwidth change together, their contributions cannot be independently attributed. IS and OS remain heavily stall-dominated; earlier layer-level analysis associates much of their traffic with repeated filter fetching in the final ResNet stage. These figures have **not** been independently validated as physical accelerator performance.

**Artifacts:** `experimental_memory_model/runs/*/config.cfg`, `experimental_memory_model/runs/*/baseline_metrics.json`, `experimental_memory_model/results.csv`, and `experimental_memory_model/runtime_settings.json`.

### Offline first-use experiment — one convolution only

This is a **single-layer study**, not a 21-layer result. The layer is `layer1.0.conv1`, GEMM **[M, N, K] = [576, 64, 576]**, using 64×64 WS and the experimental simulator. The table maintains bandwidth as the innermost configuration variable, with scheduling mode distinguished at the same bandwidth.

| Dataflow | PE array | SRAM each (KiB) | BW (words/cycle) | Prefetch mode | Total cycles | Stall cycles | Stall share |
|:--|:--|--:|--:|:--|--:|--:|--:|
| WS | 64×64 | 64 | 10 | Legacy | 265,403 | 258,510 | 97.40% |
| WS | 64×64 | 64 | 10 | Offline first-use | 32,996 | 26,103 | 79.11% |
| WS | 64×64 | 64 | 40 | Legacy | 66,758 | 59,865 | 89.68% |
| WS | 64×64 | 512 | 10 | Legacy | 32,931 | 26,038 | 79.07% |
| WS | 64×64 | 512 | 40 | Legacy | 8,356 | 1,463 | 17.51% |

All five cases have **6,893 nonstall cycles**. Holding SRAM at 64 KiB and bandwidth at 10 words/cycle, offline first-use reduces cycles by **8.04×** relative to legacy ordering and reduces modeled IFMAP DRAM request slots from **2,656,800** to **336,200**. This measures the sensitivity to a future-informed schedule; it is **not an implementable hardware-speedup claim**. The single-layer results should be archived under `experimental_memory_model/offline_first_use/` after source logs and configurations are inventoried.

### Runtime configuration and provenance

| Environment variable | Code default | Intended full-model value | Purpose |
|:--|:--|:--|:--|
| `SCALESIM_EXACT_ACTIVE_WINDOW` | `1` | `1` | Exact active-window tracking |
| `SCALESIM_WS_IFMAP_PREFETCH` | `legacy` | `legacy` | Native ordering for all seven full-model runs |
| `SCALESIM_WORD_SIZE_BYTES` | `1` | `4` | Four-byte FP32 SRAM-capacity accounting |

For the separate offline first-use experiment, set `SCALESIM_WS_IFMAP_PREFETCH=first_use`.

**Historical-environment caveat:** The metrics JSON files did not record these environment variables, and the retained log/manifest search found no explicit settings. The values above specify the **intended reproduction configuration**; they do **not** independently establish the exact environment of every historical run.

---

## Comparison requirements and limitations

- **Effective capacity differs across roots.** Upstream treats each logical stored element as one byte for capacity accounting; the intended modified runs use four bytes. A 64 KiB upstream SRAM and a 64 KiB FP32 SRAM are not element-capacity equivalent.
- **Separate one-variable ablations are required.** Different read-buffer cursors, residency accounting, word-size accounting and scheduling rules are confounded in an uncontrolled upstream-versus-modified comparison.
- **Bandwidth is per modeled transfer path.** In USER mode, the relevant `[layout] IfmapSRAMBankBandwidth` and `FilterSRAMBankBandwidth` parameters govern corresponding read paths; `[architecture_presets] Bandwidth` governs the output backing path. A setting of BW160 does not necessarily model one shared 160-word/cycle physical DRAM interface.
- **Scope excludes non-GEMM operations and physical integration.** BatchNorm, ReLU, residual addition, pooling, full-network pipelining, compressed-weight reconstruction, injected faults, fault recovery and LoRA computation are absent from the result tables unless specifically stated otherwise.
- **Patch verification is limited.** Reverse-apply checks succeeded against the two local checkout states. A clean reconstruction and controlled matched rerun remain necessary to verify the reported results independently.
- **Machine-specific paths appear in archived JSON.** Use the version-controlled configurations and `experiments/workloads/original_gemm.csv` instead of the original absolute paths when reproducing runs.

**Next validation:** Match effective SRAM capacity, per-path bandwidth, PE array, workload and dataflow, then change only one memory-model mechanism per run. Record the actual environment, backing-memory request count and per-layer stall breakdown for every new experiment.
