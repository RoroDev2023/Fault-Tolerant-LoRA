"""Apply the predeclared Part 7 rate-selection rule to the TinyBERT sweeps (selection split only)."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fault_lora.evaluation.tinybert_lora_proposal import main


if __name__ == "__main__":
    main(PROJECT_ROOT)
