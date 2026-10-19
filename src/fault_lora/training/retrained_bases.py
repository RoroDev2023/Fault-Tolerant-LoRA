"""Recovery bases built from the codebook-retrained encoding (Step 11a) and Step 11b patterns.

Used by fault_lora.training.recovery for base_kind "retrained-clustered" and
"retrained-faulty". For these kinds the recovery configuration's
source_sweep_results / source_audit_results name the Step 11b sweep and audit.
Everything is verified through the Step 11b provenance chain before use.
"""

import json

import torch

from fault_lora.evaluation.clustered_baseline import sha256_file
from fault_lora.evaluation.fault_checks import require
from fault_lora.evaluation.retrained_faults import RetrainedSweepConfig, extended_plan, load_baselines
from fault_lora.evaluation.source_hashes import compare_source_hashes
from fault_lora.memory.clustering import decode_state
from fault_lora.memory.faults import apply_fault_pattern, encoding_fingerprint, summarize_fault_pattern


RETRAINED_KINDS = ("retrained-clustered", "retrained-faulty")
SPLITS = ("validation", "test")


def prepare_retrained_sources(root, config):
    """Same return layout as recovery.prepare_sources, for the retrained encoding."""
    directory = root / config.source_sweep_results
    require(json.loads((directory / "status.json").read_text())["status"] == "complete" and json.loads((directory / "verification.json").read_text())["status"] == "passed",
            "Step 11b sweep is incomplete or unverified")
    sweep_config = RetrainedSweepConfig(**json.loads((directory / "config.json").read_text()))
    sweep_config.validate()
    require(sweep_config.audit_results == config.source_audit_results, "Recovery and Step 11b sweep use different audits")
    sources = load_baselines(root, sweep_config)
    encoding = sources["retrained"]
    fingerprint = encoding_fingerprint(encoding)
    plan = json.loads((directory / "plan.json").read_text())
    require(plan == extended_plan(sources["step7_plan"], sweep_config.extra_probabilities, sweep_config.extra_first_seed, sweep_config.patterns_per_rate), "Saved Step 11b plan changed")
    records = json.loads((directory / "measurements.json").read_text())
    require([record["pattern_id"] for record in records] == [entry["pattern_id"] for entry in plan] and plan[0]["probability"] == 0, "Step 11b records are incomplete or reordered")
    zero = records[0]["models"]["retrained"]
    saved_references = json.loads((directory / "references.json").read_text())
    require(all(saved_references["clean"][split]["num_correct"] == sources["clean_metrics"][split]["num_correct"] for split in SPLITS), "Step 11b clean replay differs from Step 3")
    references = {split: {"clean": saved_references["clean"][split], "clustered_no_faults": zero[split]} for split in SPLITS}
    paths, hashes = dict(sources["paths"]), dict(sources["hashes"])
    faulty = config.base_kind == "retrained-faulty"
    if faulty:
        matches = [record for record in records if record["pattern_id"] == config.pattern_id]
        require(len(matches) == 1 and matches[0]["probability"] > 0, "Recovery requires a saved nonzero Step 11b pattern")
        saved = matches[0]
        require(saved == json.loads((directory / "patterns" / f"{config.pattern_id}.json").read_text()), "Individual and combined Step 11b records disagree")
        paths["selected_pattern"], hashes["selected_pattern"] = root / saved["pattern"], saved["pattern_sha256"]
        require(sha256_file(paths["selected_pattern"]) == hashes["selected_pattern"], "Selected Step 11b pattern file changed")
        pattern = torch.load(paths["selected_pattern"], map_location="cpu", weights_only=True)
        require(pattern["source_fingerprint"] == fingerprint and pattern["seed"] == saved["seed"] and pattern["fault_probability"] == saved["probability"], "Selected pattern is bound to another encoding or seed")
        require({key: value for key, value in summarize_fault_pattern(encoding, pattern).items() if key != "layers"} == saved["counts"], "Selected pattern counts differ from Step 11b")
        record = {"pattern_id": config.pattern_id, "seed": saved["seed"], "probability": saved["probability"], "counts": saved["counts"], "measurements": saved["models"]["retrained"],
                  "pattern": saved["pattern"], "pattern_sha256": saved["pattern_sha256"]}
    else:
        record = {"pattern_id": None, "seed": None, "probability": 0.0, "counts": {"actual_corrupted_cells": 0, "realized_fault_rate": 0.0}, "measurements": zero}
    accepted = compare_source_hashes(root, json.loads((directory / "source_sha256.json").read_text()), "Previously used Step 11b sweep code changed")
    for filename in ["config.json", "plan.json", "status.json", "verification.json", "manifest.json", "measurements.json", "references.json", "source_sha256.json"] \
            + ([f"patterns/{config.pattern_id}.json"] if faulty else []):
        paths[f"retrained_sweep_{filename}"], hashes[f"retrained_sweep_{filename}"] = directory / filename, sha256_file(directory / filename)
    audit = root / config.source_audit_results
    for filename in ("report.json", "status.json"):
        paths[f"retrained_audit_{filename}"], hashes[f"retrained_audit_{filename}"] = audit / filename, sha256_file(audit / filename)
    require(config.clean_checkpoint_sha256 in (None, hashes["clean"]), "Clean checkpoint differs from the configured SHA-256")
    metadata = {name: value for name, value in sources["metadata"].items() if name != "model_state_dict"}
    return encoding, metadata, sources["split"], {**record, "accepted_source_hash_changes": accepted}, references, paths, hashes


def build_retrained_state(config, encoding, paths):
    """retrained-clustered: decoded retrained encoding; retrained-faulty: read through the saved pattern."""
    if config.base_kind == "retrained-clustered":
        return decode_state(encoding)
    require(config.base_kind == "retrained-faulty", "Unsupported retrained base_kind")
    return apply_fault_pattern(encoding, torch.load(paths["selected_pattern"], map_location="cpu", weights_only=True))
