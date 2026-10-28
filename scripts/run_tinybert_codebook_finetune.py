"""Step 12a: fine-tune only TinyBERT codebook values (indices fixed); no faults, LoRA, or downloads."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fault_lora.training.tinybert_codebook_finetune import main


if __name__ == "__main__":
    main(PROJECT_ROOT)
