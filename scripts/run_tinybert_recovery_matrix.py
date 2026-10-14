"""Run a declared TinyBERT LoRA recovery matrix and its transfer evaluations from --config."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fault_lora.training.tinybert_recovery_matrix import main


if __name__ == "__main__":
    main(PROJECT_ROOT)
