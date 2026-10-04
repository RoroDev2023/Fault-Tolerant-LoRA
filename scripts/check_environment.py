"""Check installed libraries and tensor devices without models or datasets."""

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()

    if sys.version_info < (3, 10):
        parser.error("Python 3.10 or later is required.")

    libraries = {
        "torch": "torch",
        "torchvision": "torchvision",
        "numpy": "numpy",
        "scikit-learn": "sklearn",
        "matplotlib": "matplotlib",
    }
    with tempfile.TemporaryDirectory(prefix="fault-lora-matplotlib-") as cache_dir:
        os.environ.setdefault("MPLCONFIGDIR", cache_dir)
        os.environ.setdefault("XDG_CACHE_HOME", cache_dir)
        modules = {name: importlib.import_module(module) for name, module in libraries.items()}
        modules["matplotlib"].use("Agg")
        importlib.import_module("matplotlib.backends.backend_agg")

        torch = modules["torch"]
        numpy = modules["numpy"]
        available = {
            "cpu": True,
            "mps": torch.backends.mps.is_available(),
            "cuda": torch.cuda.is_available(),
        }
        device = args.device
        if device == "auto":
            device = next(name for name in ("cuda", "mps", "cpu") if available[name])
        if not available[device]:
            parser.error(f"Requested device {device!r} is unavailable in this process.")

        # Verify binary-library interoperability and a tiny float32 operation.
        # These fixed arrays are installation checks, not research measurements.
        array = numpy.array([[1, 2], [3, 4]], dtype=numpy.float32)
        tensor = torch.from_numpy(array)
        numpy.testing.assert_array_equal(tensor.numpy(), array)
        expected = torch.tensor([[7, 10], [15, 22]], dtype=torch.float32)
        torch.testing.assert_close(tensor @ tensor, expected)
        selected_tensor = tensor.to(device)
        torch.testing.assert_close((selected_tensor @ selected_tensor).cpu(), expected)

        # Exercise a compiled torchvision operator to detect incompatible wheels.
        boxes = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.float32)
        scores = torch.tensor([1, 0.5], dtype=torch.float32)
        kept = modules["torchvision"].ops.nms(boxes, scores, 0.5)
        torch.testing.assert_close(kept, torch.tensor([0]))

        report = {
            "python": platform.python_version(),
            "executable": sys.executable,
            "virtual_environment": sys.prefix != sys.base_prefix,
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "cpu_count": os.cpu_count(),
            "packages": {name: importlib.metadata.version(name) for name in libraries},
            "mps_built": torch.backends.mps.is_built(),
            "available_devices": available,
            "selected_device": device,
            "checks": {
                "library_imports": "passed",
                "matplotlib_agg_import": "passed",
                "numpy_torch_bridge": "passed",
                "cpu_float32_matmul": "passed",
                "selected_device_float32_matmul": "passed",
                "torchvision_cpu_nms": "passed",
            },
        }
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
