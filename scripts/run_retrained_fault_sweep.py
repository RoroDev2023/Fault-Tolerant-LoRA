"""Step 11b: identical faults on the Step 4 and retrained encodings; evaluation only, no training."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fault_lora.evaluation.retrained_faults import main_sweep


if __name__ == "__main__":
    main_sweep(PROJECT_ROOT)
