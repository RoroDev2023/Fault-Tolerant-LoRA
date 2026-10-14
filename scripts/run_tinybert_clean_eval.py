"""Re-evaluate the Step 10a TinyBERT checkpoint on the selection split and both test sets."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fault_lora.evaluation.tinybert_clean_eval import main


if __name__ == "__main__":
    main(PROJECT_ROOT)
