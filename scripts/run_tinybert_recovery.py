"""Run one TinyBERT LoRA recovery (clean, clustered, or faulty base) from --config."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fault_lora.training.tinybert_recovery import main


if __name__ == "__main__":
    main(PROJECT_ROOT)
