"""Tests for parsing SCALE-Sim v3 compute reports."""

from pathlib import Path
import tempfile
import unittest

from hardware_sim.scalesim.run_gemm import report_rows, tally


class ScalesimReportTests(unittest.TestCase):
    def test_tally_reads_prefetch_inclusive_cycles_from_v3_report(self):
        report = (
            "LayerID, Total Cycles (incl. prefetch), Total Cycles, Stall Cycles,\n"
            "0, 12, 8, 3,\n"
            "1, 21, 17, 4,\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "COMPUTE_REPORT.csv"
            path.write_text(report)

            totals = tally(report_rows(path))

        self.assertEqual(totals, (25, 33))


if __name__ == "__main__":
    unittest.main()
