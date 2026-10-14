"""Run a Step 10c TinyBERT condition fault sweep (A or B, from --config)."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fault_lora.evaluation.tinybert_condition_sweep import main


if __name__ == "__main__":
    main(PROJECT_ROOT)
