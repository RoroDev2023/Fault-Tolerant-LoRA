"""Tests for generated SCALE-Sim v3 architecture configs."""

import configparser
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "hardware_sim"))

from scalesim.make_fixed_bw_config import build_config


class ScalesimConfigTests(unittest.TestCase):
    def test_builds_v3_config_schema_and_selected_hardware_parameters(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = build_config(16, 16, 10, out_path=Path(temporary) / "array.cfg")
            parser = configparser.ConfigParser()
            self.assertTrue(parser.read(path))

        self.assertEqual(
            parser.sections(),
            ["general", "architecture_presets", "layout", "sparsity", "run_presets"],
        )
        self.assertEqual(parser.get("general", "run_name"), "fault_lora_16x16_os_user_bw")
        self.assertEqual(parser.getint("architecture_presets", "arrayheight"), 16)
        self.assertEqual(parser.getint("architecture_presets", "arraywidth"), 16)
        self.assertEqual(parser.get("architecture_presets", "dataflow"), "os")
        self.assertEqual(parser.getint("architecture_presets", "bandwidth"), 10)
        self.assertEqual(parser.get("run_presets", "interfacebandwidth"), "USER")
        self.assertFalse(parser.getboolean("run_presets", "useramulatortrace"))
        self.assertFalse(parser.getboolean("sparsity", "sparsitysupport"))

    def test_rejects_nonpositive_architecture_and_invalid_dataflow(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "invalid.cfg"
            for arguments in (
                (0, 16, 10, "os"),
                (16, 16, 0, "os"),
                (16, 16, 10, "invalid"),
            ):
                with self.subTest(arguments=arguments):
                    with self.assertRaises(ValueError):
                        build_config(
                            arguments[0],
                            arguments[1],
                            arguments[2],
                            dataflow=arguments[3],
                            out_path=output,
                        )
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
