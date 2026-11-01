"""Tests for parsing SCALE-Sim v3 compute reports."""

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hardware_sim.scalesim.run_gemm import (
    DEFAULT_ARRAY_SIZES,
    collect_layer_metrics,
    parse_array_sizes,
    report_rows,
    run_single_case,
    summarize_layer_metrics,
    tally,
    write_layer_metrics,
)


class ScalesimReportTests(unittest.TestCase):
    def test_tally_reads_prefetch_inclusive_cycles_from_v3_report(self):
        report = (
            "LayerID, Total Cycles (incl. prefetch), Total Cycles, Stall Cycles, "
            "Overall Util %, Mapping Efficiency %, Compute Util %,\n"
            "0, 12, 8, 3, 60, 75, 50,\n"
            "1, 21, 17, 4, 70, 50, 25,\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "COMPUTE_REPORT.csv"
            path.write_text(report)

            totals = tally(report_rows(path))

        self.assertEqual(totals, (25, 33))

    def test_collects_layer_metrics_and_summarizes_inactive_pe_equivalents(self):
        with tempfile.TemporaryDirectory() as temporary:
            workload = Path(temporary) / "workload.csv"
            workload.write_text("Layer name,M,N,K,Sparsity,\nconv1,2,3,4,1:1,\nfc,1,2,3,1:1,\n")
            rows = [
                {"LayerID": "0", "Mapping Efficiency %": "75", "Compute Util %": "50"},
                {"LayerID": "1", "Mapping Efficiency %": "50", "Compute Util %": "25"},
            ]

            metrics = collect_layer_metrics(rows, workload, 16, "baseline", "base")
            summary = summarize_layer_metrics(metrics)

            output = Path(temporary) / "layer_metrics.csv"
            write_layer_metrics(output, [{"layer_metrics": metrics}])
            with output.open(newline="") as stream:
                saved_rows = list(csv.DictReader(stream))

        self.assertEqual([row["layer_name"] for row in metrics], ["conv1", "fc"])
        self.assertEqual([row["inactive_pe_equivalents"] for row in metrics], [128, 192])
        self.assertEqual(summary["mean_mapping_efficiency_percent"], 62.5)
        self.assertEqual(summary["mean_compute_utilization_percent"], 37.5)
        self.assertEqual(summary["mean_inactive_pe_equivalents"], 160)
        self.assertEqual(len(saved_rows), 2)

    def test_metric_collection_rejects_missing_metrics_or_workload_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            workload = Path(temporary) / "workload.csv"
            workload.write_text("Layer name,M,N,K,Sparsity,\nconv1,2,3,4,1:1,\n")
            with self.assertRaisesRegex(ValueError, "Missing or invalid"):
                collect_layer_metrics([{"LayerID": "0"}], workload, 16, "baseline", "base")
            with self.assertRaisesRegex(ValueError, "has 0 layers"):
                collect_layer_metrics([], workload, 16, "baseline", "base")

    def test_default_array_sizes_include_32_and_custom_sizes_are_parsed(self):
        self.assertEqual(DEFAULT_ARRAY_SIZES, (16, 32, 64, 128))
        self.assertEqual(parse_array_sizes(None), (16, 32, 64, 128))
        self.assertEqual(parse_array_sizes("16,32,64,128"), (16, 32, 64, 128))

    def test_clustered_lora_case_combines_base_and_adapter_metrics(self):
        report_data = {
            "original_gemm.csv": (
                "LayerID, Total Cycles (incl. prefetch), Total Cycles, Stall Cycles, "
                "Overall Util %, Mapping Efficiency %, Compute Util %,\n"
                "0, 12, 8, 3, 60, 75, 50,\n"
                "1, 21, 17, 4, 70, 50, 25,\n"
            ),
            "lora_gemm.csv": (
                "LayerID, Total Cycles (incl. prefetch), Total Cycles, Stall Cycles, "
                "Overall Util %, Mapping Efficiency %, Compute Util %,\n"
                "0, 4, 3, 1, 40, 25, 10,\n"
            ),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workloads = root / "workloads"
            workloads.mkdir()
            (workloads / "original_gemm.csv").write_text(
                "Layer name,M,N,K,Sparsity,\nconv1,2,3,4,1:1,\nfc,1,2,3,1:1,\n"
            )
            (workloads / "lora_gemm.csv").write_text(
                "Layer name,M,N,K,Sparsity,\nconv1.lora_A,2,3,4,1:1,\n"
            )

            def execute(_config, workload, out_dir, _layout):
                out_dir.mkdir(parents=True)
                report = out_dir / "COMPUTE_REPORT.csv"
                report.write_text(report_data[workload.name])
                return report

            with (
                patch("hardware_sim.scalesim.run_gemm.ensure_scalesim"),
                patch("hardware_sim.scalesim.run_gemm.validate_config"),
                patch("hardware_sim.scalesim.run_gemm.maybe_execute_simulation", side_effect=execute),
            ):
                result = run_single_case(
                    16,
                    "clustered_lora",
                    root / "scale_16x16_user_bw.cfg",
                    workloads,
                    root / "layout.csv",
                    root / "results",
                    False,
                    10,
                )

        self.assertEqual(result["total_cycles"], 28)
        self.assertEqual(result["prefetch_cycles"], 37)
        self.assertEqual(len(result["layer_metrics"]), 3)
        self.assertEqual(
            [row["workload_component"] for row in result["layer_metrics"]],
            ["base", "base", "lora"],
        )
        self.assertAlmostEqual(result["mean_mapping_efficiency_percent"], 50)
        self.assertAlmostEqual(result["mean_compute_utilization_percent"], 28.3333333333)
        self.assertAlmostEqual(result["mean_inactive_pe_equivalents"], 183.4666666667)


if __name__ == "__main__":
    unittest.main()
