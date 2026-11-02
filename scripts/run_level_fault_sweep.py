"""Step 13b: level-dependent faults on the retrained ResNet-18, sorted vs frequent-at-0 mapping; no repair and BN recalibration."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fault_lora.evaluation.level_fault_sweep import main


if __name__ == "__main__":
    main(PROJECT_ROOT)
