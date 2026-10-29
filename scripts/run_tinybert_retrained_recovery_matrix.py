"""Step 12c: TinyBERT LoRA on the codebook-retrained baseline (two adapter placements), then transfer evaluations."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fault_lora.training.tinybert_retrained_recovery import main


if __name__ == "__main__":
    main(PROJECT_ROOT)
