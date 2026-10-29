"""Tests for LoRA adapter GEMM topology export."""

import csv
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "hardware_sim"))

from scalesim.extract_lora_gemm import collect_lora_gemm_layers, write_gemm_topology


class ScalesimLoRAWorkloadTests(unittest.TestCase):
    def test_collects_a_and_b_gemms_for_conv_and_linear_targets(self):
        model = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4, 2),
        )

        layers = collect_lora_gemm_layers(model, image_size=8, rank=2)

        self.assertEqual(
            [(layer["name"], layer["M"], layer["N"], layer["K"]) for layer in layers],
            [
                ("0.lora_A", 16, 2, 27),
                ("0.lora_B", 16, 4, 2),
                ("3.lora_A", 1, 2, 4),
                ("3.lora_B", 1, 2, 2),
            ],
        )
        self.assertTrue(all(int(layer["macs"]) == layer["M"] * layer["N"] * layer["K"] for layer in layers))

    def test_writes_scalesim_v3_topology_with_trailing_column(self):
        layers = [{"name": "layer.lora_A", "M": 16, "N": 2, "K": 27, "macs": 864}]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lora_gemm.csv"
            write_gemm_topology(layers, path)

            with path.open(newline="") as stream:
                rows = list(csv.reader(stream))

        self.assertEqual(rows[0], ["Layer name", "M", "N", "K", "Sparsity", ""])
        self.assertEqual(rows[1], ["layer.lora_A", "16", "2", "27", "1:1", ""])

    def test_rejects_empty_topology(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                write_gemm_topology([], Path(temporary) / "empty.csv")


if __name__ == "__main__":
    unittest.main()
