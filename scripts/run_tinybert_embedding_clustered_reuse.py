"""Run Step 10c condition B clustering: Step 10b Linear records reused, embedding tables clustered."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fault_lora.evaluation.tinybert_embedding_clustered_reuse import main


if __name__ == "__main__":
    main(PROJECT_ROOT)
