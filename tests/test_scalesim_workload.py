"""Shape and serialization tests for the SCALE-Sim GEMM workload exporter."""

import csv
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "hardware_sim"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scalesim.extract_resnet_gemm import collect_gemm_layers, write_gemm_topology
from fault_lora.models.resnet import build_resnet18


class ScalesimWorkloadTests(unittest.TestCase):
    def test_maps_convolution_and_linear_to_gemm_dimensions(self):
        model = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4, 2),
        )
        layers = collect_gemm_layers(model, image_size=8)

        self.assertEqual(
            [(layer["name"], layer["M"], layer["N"], layer["K"]) for layer in layers],
            [("0", 16, 4, 27), ("3", 1, 2, 4)],
        )

    def test_writes_sparsity_and_trailing_empty_column_required_by_scalesim(self):
        layers = [{"name": "conv", "M": 16, "N": 4, "K": 27}]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "original_gemm.csv"
            write_gemm_topology(layers, output)

            with output.open(newline="") as stream:
                rows = list(csv.reader(stream))

        self.assertEqual(rows[0], ["Layer name", "M", "N", "K", "Sparsity", ""])
        self.assertEqual(rows[1], ["conv", "16", "4", "27", "1:1", ""])

    def test_rejects_grouped_convolution_instead_of_emitting_wrong_gemm(self):
        model = nn.Conv2d(3, 3, kernel_size=3, groups=3)
        with self.assertRaisesRegex(ValueError, "groups=1"):
            collect_gemm_layers(model, image_size=8)

    def test_project_resnet18_exports_all_convolution_and_classifier_gemms(self):
        model = build_resnet18(num_classes=10, pretrained=False)
        layers = collect_gemm_layers(model, image_size=96)

        self.assertEqual(len(layers), 21)
        self.assertEqual((layers[0]["name"], layers[0]["M"], layers[0]["N"], layers[0]["K"]),
                         ("conv1", 2304, 64, 147))
        self.assertEqual((layers[-1]["name"], layers[-1]["M"], layers[-1]["N"], layers[-1]["K"]),
                         ("fc", 1, 10, 512))


if __name__ == "__main__":
    unittest.main()
