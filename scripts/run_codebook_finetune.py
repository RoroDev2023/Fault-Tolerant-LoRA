"""Step 11a: fine-tune only the codebook values of the clustered ResNet-18; no faults or downloads."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fault_lora.training.codebook_finetune import main


if __name__ == "__main__":
    main(PROJECT_ROOT)
