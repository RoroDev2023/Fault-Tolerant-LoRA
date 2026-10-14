"""Run the Step 10c condition B fault-injection audit."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fault_lora.evaluation.tinybert_condition_audit import main


if __name__ == "__main__":
    main(PROJECT_ROOT)
