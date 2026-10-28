"""Record Slurm job provenance beside the artifacts of a completed model run."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="effective config used by the job")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    config = json.loads(args.config.read_text())
    for key in ("run_id", "results_root"):
        if key not in config:
            parser.error(f"Effective config is missing {key!r}: {args.config}")

    results_root = Path(config["results_root"])
    if not results_root.is_absolute():
        results_root = repo_root / results_root
    result_dir = results_root / config["run_id"]
    if not result_dir.is_dir():
        parser.error(f"Expected completed run directory does not exist: {result_dir}")

    revision = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    metadata = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_revision": revision,
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "cluster": os.environ.get("SLURM_CLUSTER_NAME"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "account": os.environ.get("SLURM_JOB_ACCOUNT"),
            "nodes": os.environ.get("SLURM_JOB_NODELIST"),
        },
    }

    effective_config = result_dir / "config.hpc.json"
    temporary_config = effective_config.with_suffix(".json.tmp")
    temporary_config.write_text(json.dumps(config, indent=2) + "\n")
    temporary_config.replace(effective_config)
    output = result_dir / "hpc_job.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(output)
    print(f"Recorded Slurm provenance: {output}")


if __name__ == "__main__":
    main()
