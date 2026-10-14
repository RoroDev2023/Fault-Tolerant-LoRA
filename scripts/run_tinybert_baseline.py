"""Run the Step 10a TinyBERT / SST-2 clean evaluation from the project source tree."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fault_lora.evaluation.tinybert_baseline import main


if __name__ == "__main__":
    main(PROJECT_ROOT)
