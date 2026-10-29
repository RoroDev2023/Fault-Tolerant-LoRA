"""Step 12c: analysis of LoRA on the retrained TinyBERT baseline, Linear vs Linear + embedding adapters; reads saved runs only."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fault_lora.evaluation.tinybert_retrained_lora_controls import main


if __name__ == "__main__":
    main(PROJECT_ROOT)
