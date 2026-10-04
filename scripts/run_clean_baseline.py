"""Run the Step 3 CIFAR-10 baseline from the project source tree."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fault_lora.training.baseline import main


if __name__ == "__main__":
    main(PROJECT_ROOT)
