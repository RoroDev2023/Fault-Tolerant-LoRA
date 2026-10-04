"""Run Step 5 using existing local encodings; CPU only, no data downloads."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fault_lora.evaluation.fault_simulation import main


if __name__ == "__main__":
    main(PROJECT_ROOT)
