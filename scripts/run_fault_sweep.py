"""Run Step 7 with audited local artifacts and fixed evaluation splits."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fault_lora.evaluation.fault_sweep import main


if __name__ == "__main__":
    main(PROJECT_ROOT)
