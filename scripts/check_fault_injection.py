"""Run Step 6 on local ResNet artifacts; CPU only, no dataset evaluation."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fault_lora.evaluation.fault_checks import main


if __name__ == "__main__":
    main(PROJECT_ROOT)
