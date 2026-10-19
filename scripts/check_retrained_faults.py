"""Step 11b: eight-check audit of the codebook-retrained encoding; CPU, no dataset."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fault_lora.evaluation.retrained_faults import main_audit


if __name__ == "__main__":
    main_audit(PROJECT_ROOT)
