# ResNet-18 → SCALE-Sim GEMM baseline integration

This package is designed to live under `hardware_sim/scalesim/` on a local
`feat/resnet-scalesim` branch of Fault-Tolerant-LoRA.

## What is verified

- Imports the real project functions `fault_lora.models.resnet.build_resnet18` and
  `fault_lora.models.lora.attach_lora` and the exact `LoRALayer` modules.
- Uses a **single input** of shape `(1, 3, 96, 96)` per the research README.
- Preserves the original ResNet-18 7×7 stride-2 stem and max-pooling.
- Selects all 21 Conv2d/Linear modules (including three downsample shortcuts).
- Extracts runtime base, LoRA A, and LoRA B GEMM-equivalent dimensions by forward hooks.
- Uses rank=4, alpha=4, with no pretrained download/checkpoint, dataset or training.
- Exports `original_gemm.csv`, `lora_gemm.csv`, and `manifest.json`.
- The simulation runner derives the three **requested scenarios**, but scenario 2
  and 3 are identical until a fault-aware memory extension is implemented.

## Required Python environments

Two separate environments may be necessary:
1. Python: `torch`, `torchvision` installed.
2. SCALE-Sim editable virtual environment: `scalesim` installed.

Start in the **Fault-Tolerant-LoRA root**, with `scale-sim-v3` as a sibling:

```sh
# Export shapes (model code runs without downloading pretrained weights)
python hardware_sim/scalesim/extract_resnet_gemm.py

# Run the full preliminary GEMM-only simulation
../scale-sim-v3/.venv/bin/python hardware_sim/scalesim/run_gemm.py \
  --config ../scale-sim-v3/configs/scale.cfg \
  --layout ../scale-sim-v3/layouts/conv_nets/test.csv

# Inspect outputs
cat results/hardware_sim/scalesim/comparison.csv
head -n 8 results/hardware_sim/scalesim/layer_comparison.csv
```

If `python` doesn't import `torch`/`torchvision`, use local
`.venv/bin/python` for the **export** command, assuming its environment has
been installed. If SCALE-Sim fails to import `scalesim` in its own venv, revisit
its editable installation.

`run_gemm.py` creates reports in the research repo's ignored `results/` directory.
The generated workload CSV/JSON files are ignored using this subdirectory's
`.gitignore`; keep scripts and documentation under version control.

## Scientific limits (do not report as final hardware results)

- Current CSV uses im2col GEMM equivalents for convolutions. This is *not* a
  cycle-accurate convolution memory movement model.
- `run_gemm.py` sums per-operation cycles serially and ignores residual branch
  overlap, BN/ReLU/pooling, LoRA output addition and pipeline effects.
- Baseline and adapter GEMMs do not simulate the codebook/index compressed
  weight representation and **do not inject memory faults**.
- In the original research code, k=16 codebooks and a `uint8` index per scalar
  weight are used in software. Ideal four-bit packing is only a storage estimate.
  Codebooks, biases, BN data, and adapters are assumed reliable in the fault model.
- Memory corruption in the model is a persistent adjacent-level index-cell
  perturbation, not an automatic memory latency penalty.
- The three scenarios are comparison labels rather than distinct compressed
  memory simulations until a custom extension is implemented.

## Next validated implementation step

Read the code under `src/fault_lora/memory/` before building custom timing. Specify
where the index is fetched, where centroids are stored, whether the hardware
caches decoded weights, whether the original weight exists as dense data in
memory, what bandwidth is used, and the exact reason (if any) faults add cycles.
Only then modify SCALE-Sim's memory service paths and test new timing behavior.


