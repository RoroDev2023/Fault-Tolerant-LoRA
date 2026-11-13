# ResNet-18 on a 128×128 array: SCALE-Sim cycle check

Why one ResNet-18 inference (96×96 input, 10 classes) comes out at about
4 million cycles on a 128×128 weight-stationary array, and how much of that
is real.

**Short answer:** the ResNet-to-GEMM mapping is correct. Most of the cycles
come from SCALE-Sim reading the same weights from DRAM many times when a
layer's weights do not fit in the 64 KB filter SRAM. With the weights read
once, the slowest layer drops from 1,141,106 cycles to 56,303.

## Files

| File | Purpose |
| --- | --- |
| `scale_128x128_calc.cfg` | 128×128, 64 KB SRAMs, CALC bandwidth (no memory stalls) |
| `scale_128x128_user_bank160.cfg` | Same, USER mode, 160 words/cycle for ifmap and filter reads |
| `scale_128x128_user_bw10.cfg` | Same, USER mode, 10 words/cycle |
| `summarize_bench.py` | Splits `run_gemm.py` totals into compute and stall cycles, per run and per layer |
| `single_layer_sram_test.py` | Runs one layer with different filter SRAM sizes and bandwidths and reports how many times its weights were read from DRAM |

All configs copy `../scale_32x32_user_bw.cfg` with the array set to 128×128.
SCALE-Sim words are 1 byte, so "KB" of SRAM is thousands of weights.

## How to run

SCALE-Sim v3 was used at upstream commit `f79d7bc`; this repository does not
pin a version.

```sh
# full network, one config per run
python hardware_sim/scalesim/run_gemm.py --config hardware_sim/scalesim/benchmark/scale_128x128_calc.cfg \
    --layout ../scale-sim-v3/layouts/conv_nets/test.csv --out results/hardware_sim/scalesim_bench_128/calc
python hardware_sim/scalesim/benchmark/summarize_bench.py --layers-for calc

# one layer, 64 KB vs 4 MB filter SRAM (about 15 minutes, mostly the 64 KB run)
PYTHONPATH=../scale-sim-v3 python hardware_sim/scalesim/benchmark/single_layer_sram_test.py \
    --layout ../scale-sim-v3/layouts/conv_nets/test.csv
```

## Results

### Whole network

| Setting | Base | LoRA (rank 4) | Total | Stall share |
| --- | ---: | ---: | ---: | ---: |
| CALC (compute only) | 297,097 | 141,711 | 438,808 | 0% |
| USER, 160 words/cycle | 4,091,466 | 143,321 | 4,234,787 | 89.6% |

The compute-only cycles match the weight-stationary formula
ceil(K/128) · ceil(N/128) · (2·128 + 128 + M − 2) layer by layer, so the GEMM
shapes and the array mapping are right. The USER run at 10 words/cycle did not
finish within 3.7 hours.

### Where the stalls come from (USER, 160 words/cycle, 64 KB filter SRAM)

| Layer | Weights | Times the weights were read from DRAM | Cycles | Stall share |
| --- | ---: | ---: | ---: | ---: |
| layer4.0.conv2, layer4.1.conv1, layer4.1.conv2 (each) | 2,359,296 | 77 | 1,141,106 | 95% |
| layer4.0.conv1 | 1,179,648 | 37 | 274,301 | 90% |
| layer3.0.conv2, layer3.1.conv1, layer3.1.conv2 (each) | 589,824 | 28 | 104,121 | 86% |
| layer3.0.conv1 | 294,912 | 12 | 22,749 | 67% |

"Times read" is the DETAILED_ACCESS_REPORT's DRAM Filter Reads divided by the
layer's weight count. The three largest layers alone are 3.4 M of the 4.09 M
base cycles, while each needs only 56,303 cycles of compute.

### Single-layer test: layer4.0.conv2 (M=9, N=512, K=4608)

| Variant | Filter SRAM | Total cycles | Stall cycles | DRAM filter reads |
| --- | ---: | ---: | ---: | ---: |
| CALC | 64 KB | 56,303 | 0 | 2,359,296 (1.0×) |
| USER, 160 words/cycle | 64 KB | 1,141,106 | 1,084,803 | 181,679,200 (77×) |
| USER, 160 words/cycle | 2,304 KB (exactly the weights) | 57,518 | 1,215 | 4,718,720 (2.0×) |
| USER, 160 words/cycle | 4,096 KB | 56,303 | 0 | 4,194,560 (1.8×) |
| USER, 10 words/cycle | 2,304 KB | 471,859 | 415,556 | 4,718,600 (2.0×) |
| USER, 10 words/cycle | 4,096 KB | 375,637 | 319,334 | 4,194,320 (1.8×) |

With the weights in SRAM the stalls disappear at 160 words/cycle. SCALE-Sim
still reads 1.8–2× the weights: with 4,096 KB it fills the whole buffer
(4,194,304 words) at the start, and with 2,304 KB it reads the weights twice.
At 10 words/cycle that extra traffic shows: streaming the weights once takes
235,930 cycles (2,359,296 / 10), while SCALE-Sim reports 375,637 to 471,859.

## Why the weights are re-read

SCALE-Sim's read buffer (`scalesim/memory/read_buffer.py`) splits the operand
into lines of total_size / 100 words and keeps a sliding window of them in the
order it expects to use them. When a requested address is outside the window,
it keeps prefetching new lines, wrapping around the operand if needed, until
the address is inside. Each wrap reads the whole layer's weights from DRAM
again. With a 64 KB buffer and 2.3 MB of weights this happens about once every
two of the layer's 144 weight tiles. How often it happens grows with how far
the weights overflow the buffer: 12× at 4.5 times the buffer, 77× at 36 times.

In a weight-stationary GEMM every 128×128 weight tile is loaded once and used
for all M rows, so each weight is needed exactly once. A double-buffered 64 KB
filter SRAM is enough to stream them (one tile is 16 KB). The repeated reads
are a modelling artifact, not a property of the design.

## Bandwidth setting

In USER mode SCALE-Sim takes ifmap and filter DRAM bandwidth from
`[layout] IfmapSRAMBankBandwidth` and `FilterSRAMBankBandwidth`
(`single_layer_sim.py`); `[architecture_presets] Bandwidth` only sets the ofmap
write bandwidth. `../make_fixed_bw_config.py` sets only `Bandwidth`, so a
config made with it does not limit weight and input reads to the requested
words per cycle. The bank bandwidth must also be divisible by the bank count
(10 here).

## What this means for the cycle count

* With 160 words/cycle and weights read once, the large layers are compute
  bound, and the network total is close to the compute-only 438,808 cycles.
* At 10 words/cycle, reading all 11.2 M weights once takes about 1.1 M cycles,
  so that is the floor for one inference at that bandwidth. Even with the
  weights in SRAM, SCALE-Sim puts layer4.0.conv2 at 1.6–2× its floor because
  of the extra buffer reads above.
* Numbers from SCALE-Sim runs with a 64 KB filter SRAM overstate the cycle
  count by up to 20× for the layer3 and layer4 convolutions. Size the filter
  SRAM to the largest layer (2.3 MB) to remove the artifact, and model the
  real buffer size separately.
