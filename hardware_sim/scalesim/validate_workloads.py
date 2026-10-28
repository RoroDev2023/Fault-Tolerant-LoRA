#!/usr/bin/env python3
"""Fast sanity check of exported GEMM geometry, no PyTorch required."""
import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parent / 'workloads'
manifest = json.loads((root / 'manifest.json').read_text())
with (root / 'original_gemm.csv').open(newline='') as f:
    originals = list(csv.DictReader(f))
with (root / 'lora_gemm.csv').open(newline='') as f:
    adapters = list(csv.DictReader(f))
assert manifest['image_size'] == 96
assert len(manifest['layers']) == len(originals) == 21
assert len(adapters) == 42
names = {r['name'] for r in manifest['layers']}
assert {'conv1', 'layer2.0.downsample.0', 'layer3.0.downsample.0', 'layer4.0.downsample.0', 'fc'} <= names
first = manifest['layers'][0]
assert first['name'] == 'conv1'
assert [first['base'][s] for s in ('M', 'N', 'K')] == [2304, 64, 147]
assert sum(r['base']['MACs'] for r in manifest['layers']) > 0
print('PASS: 21 base layers + 42 LoRA operations; 96x96 ImageNet-stem topology')
print('First convolution GEMM M,N,K:', *[first['base'][s] for s in ('M','N','K')])
