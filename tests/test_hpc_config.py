"""Tests for safe, non-mutating HPC config overlays."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_hpc_config import create_overlay


class HpcConfigTests(unittest.TestCase):
    def test_overlay_changes_device_and_data_root_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            output = root / "derived" / "config.json"
            original = {"device": "mps", "data_root": "data", "seed": 42}
            source.write_text(json.dumps(original))

            result = create_overlay(source, output, device="cuda", data_root="/shared/cifar")

            self.assertEqual(result, {"device": "cuda", "data_root": "/shared/cifar", "seed": 42})
            self.assertEqual(json.loads(source.read_text()), original)
            self.assertEqual(json.loads(output.read_text()), result)

    def test_overlay_rejects_missing_override_field(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            output = root / "derived.json"
            source.write_text(json.dumps({"run_id": "cpu-audit"}))

            with self.assertRaisesRegex(ValueError, "no device setting"):
                create_overlay(source, output, device="cuda")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
