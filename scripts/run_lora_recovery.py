"""Run Step 8 on saved local artifacts; no model or dataset downloads."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fault_lora.training.recovery import main


if __name__ == "__main__":
    main(PROJECT_ROOT)
