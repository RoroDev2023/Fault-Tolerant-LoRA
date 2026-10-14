"""Measure k-means clustering loss for TinyBERT with the unchanged Step 4 algorithm.

Eligible cells are every nn.Linear weight, which is what `cluster_model`
already selects. Embedding tables, LayerNorm parameters, and biases are kept
exactly as reliable untouched state.
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import time

import torch

from fault_lora.evaluation.clustered_baseline import save_torch, sha256_file, write_json
from fault_lora.evaluation.fault_checks import require, require_state_equal
from fault_lora.evaluation.tinybert_clean_eval import CleanEvalConfig, environment_record, source_artifacts, verify_source_artifacts, verify_source_code
from fault_lora.evaluation.tinybert_sources import build_tinybert, evaluate_splits, linear_weight_names, make_loaders, prepare_tinybert_sources, protected_state
from fault_lora.memory.clustering import cluster_model, decode_layer, decode_state, eligible_weight_names, physical_levels, storage_estimates
from fault_lora.memory.faults import encoding_fingerprint
from fault_lora.training.baseline import choose_device

REPRESENTATION = "per-tensor scalar codebooks and indices for every nn.Linear weight; no faults or adapters"


@dataclass
class TinyBertClusteredConfig:
    run_id: str = "tinybert-sst2-clustered-k16-seed42"
    num_clusters: int = 16
    seed: int = 42
    n_init: int = 3
    max_iter: int = 100
    tol: float = 1e-4
    clustering_threads: int = 4
    device: str = "cpu"
    clean_checkpoint_sha256: str = "eace4b3b8cb22e94151b3091b9a6b216afc1443d371cb449f68ba5d512d347f6"
    source_clean_eval_results: str = "results/tinybert-sst2-clean-twotest-seed42"
    checkpoint_root: str = "checkpoints"
    results_root: str = "results"

    def validate(self):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.run_id):
            raise ValueError("run_id must be a simple directory name")
        for name in ("num_clusters", "seed", "n_init", "max_iter", "clustering_threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if not 1 <= self.num_clusters <= 256 or not 0 <= self.seed < 2**32:
            raise ValueError("Invalid cluster count or seed")
        if min(self.n_init, self.max_iter, self.clustering_threads) < 1 or not math.isfinite(self.tol) or self.tol < 0:
            raise ValueError("Invalid KMeans configuration")
        if not isinstance(self.clean_checkpoint_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.clean_checkpoint_sha256):
            raise ValueError("clean_checkpoint_sha256 must be 64 lowercase hex characters")
        if self.device not in ("cpu", "mps", "cuda", "auto"):
            raise ValueError("Invalid device")


def prepare_clustered_sources(root, config):
    """The verified clean-evaluation run and the Step 10a sources it used."""
    clean_dir = root / config.source_clean_eval_results
    read = lambda name: json.loads((clean_dir / name).read_text())
    require(read("status.json")["status"] == "complete" and read("verification.json")["status"] == "passed", "Two-test clean evaluation is incomplete or unverified")
    clean_config = CleanEvalConfig(**read("config.json"))
    clean_config.validate()
    verify_source_artifacts(root, read("manifest.json"), "Two-test clean evaluation")
    verify_source_code(root, clean_dir, "Two-test clean evaluation")
    sources = prepare_tinybert_sources(root, clean_config.source_results, expected_original_test_examples=clean_config.expected_original_test_examples)
    require(sources.hashes["clean_checkpoint"] == config.clean_checkpoint_sha256, "Clean checkpoint differs from the configured SHA-256")
    require(read("test_sets.json") == sources.test_sets, "Test sets differ from the clean evaluation")
    clean_metrics = read("metrics.json")
    for name in ("config", "manifest", "metrics", "test_sets", "status", "verification", "source_sha256"):
        sources.paths[f"clean_eval_{name}"] = clean_dir / f"{name}.json"
        sources.hashes[f"clean_eval_{name}"] = sha256_file(clean_dir / f"{name}.json")
    return sources, clean_metrics


def run_tinybert_clustered(config, project_root):
    config.validate()
    root = Path(project_root).resolve()
    device = choose_device(config.device)
    torch.set_num_threads(config.clustering_threads)
    sources, clean_metrics = prepare_clustered_sources(root, config)
    results_dir = root / config.results_root / config.run_id
    checkpoint_dir = root / config.checkpoint_root / config.run_id
    if results_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError("Run directories already exist; choose a new run_id to preserve artifacts")
    results_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    write_json(results_dir / "config.json", asdict(config))
    write_json(results_dir / "status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
    started = time.monotonic()
    try:
        files = sorted((root / "src/fault_lora").rglob("*.py")) + [root / "scripts/run_tinybert_clustered.py", root / "requirements.txt", root / "requirements-macos-arm64.lock.txt"]
        write_json(results_dir / "source_sha256.json", {str(path.relative_to(root)): sha256_file(path) for path in files})
        environment = {**environment_record(device, config.clustering_threads, ("scikit-learn", "threadpoolctl")), "clustering_device": "cpu"}
        write_json(results_dir / "environment.json", environment)
        write_json(results_dir / "test_sets.json", sources.test_sets)
        clean = sources.model
        clean_state = {name: value.clone() for name, value in clean.state_dict().items()}
        eligible = eligible_weight_names(clean)
        require(eligible == linear_weight_names(clean), "cluster_model's eligible tensors are not exactly the nn.Linear weights")
        protected = protected_state(clean, set(eligible))
        require(protected["other"]["num_tensors"] == 0, f"Unclassified reliable tensors: {protected['other']['tensors']}")
        scope = {
            "eligible_tensor_names": eligible, "num_eligible_tensors": len(eligible),
            "eligible_weights": sum(clean_state[name].numel() for name in eligible),
            "total_parameters": sum(value.numel() for value in clean.parameters()),
            "protected": protected,
        }
        scope["eligible_fraction_of_parameters"] = scope["eligible_weights"] / scope["total_parameters"]
        scope["embedding_fraction_of_parameters"] = protected["embedding_tables"]["num_values"] / scope["total_parameters"]
        write_json(results_dir / "scope.json", scope)
        print(json.dumps({"event": "eligible_tensors", "count": len(eligible), "weights": scope["eligible_weights"]}), flush=True)

        def progress(name, statistics):
            print(json.dumps({"event": "layer_clustered", "tensor": name, **statistics}), flush=True)

        encoding, layers = cluster_model(clean, config.num_clusters, seed=config.seed, n_init=config.n_init, max_iter=config.max_iter, tol=config.tol,
                                        threads=config.clustering_threads, progress=progress)
        require_state_equal({name: value for name, value in clean.state_dict().items()}, clean_state, "Clean model after clustering")
        metadata = {
            "architecture": sources.checkpoint["architecture"], "model_config": sources.checkpoint["model_config"], "num_classes": sources.checkpoint["num_classes"],
            "classes": sources.checkpoint["classes"], "dataset": sources.checkpoint["dataset"], "preprocessing": sources.checkpoint["preprocessing"],
            "split_sha256": sources.checkpoint["split_sha256"], "test_set_sha256": {name: item["sha256"] for name, item in sources.test_sets.items()},
            "source_checkpoint": sources.metrics["checkpoint"], "source_checkpoint_sha256": sources.hashes["clean_checkpoint"],
            "clustering_config": asdict(config), "representation": REPRESENTATION,
            "scope": "every nn.Linear weight: Q, K, V, attention output, FFN intermediate and output in all layers, pooler dense, classifier",
            "reliable_state": "embedding tables (word, position, token type), LayerNorm parameters, and all biases are kept exactly and never faulted",
            "physical_mapping": "sorted codebook; identity index-to-level; one cell per index; number of levels equals actual codebook size",
            "codebook_storage_assumption": "reliable codebooks and mapping metadata", "faults_injected": 0, "environment": environment,
        }
        encoding.update(metadata)
        encoded_path = checkpoint_dir / "encoding.pt"
        save_torch(encoded_path, encoding)
        # Evaluate weights reconstructed from the saved artifact, not fit-time labels.
        saved_encoding = torch.load(encoded_path, map_location="cpu", weights_only=True)
        decoded = decode_state(saved_encoding)
        for name, record in saved_encoding["clustered_tensors"].items():
            require(torch.equal(decoded[name], decode_layer(record, levels=physical_levels(record))), f"Physical-level and index decoding disagree: {name}")
            layers[name].update({"codebook": record["codebook"].tolist(), "index_to_level": record["index_to_level"].tolist(), "level_to_index": record["level_to_index"].tolist()})
        require(set(saved_encoding["clustered_tensors"]) == set(eligible), "Saved clustered tensors differ from the eligible list")
        require_state_equal(saved_encoding["untouched_state"], {name: value for name, value in clean_state.items() if name not in eligible}, "Untouched reliable state")
        dense_path = checkpoint_dir / "clustered_dense.pt"
        save_torch(dense_path, {"model_state_dict": decoded, **metadata})
        write_json(results_dir / "layers.json", layers)
        storage = storage_estimates(saved_encoding)
        storage.update({"serialized_encoding_bytes": encoded_path.stat().st_size, "serialized_dense_checkpoint_bytes": dense_path.stat().st_size,
                        "serialized_clean_checkpoint_bytes": sources.paths["clean_checkpoint"].stat().st_size,
                        "eligible_original_tensor_bytes": sum(clean_state[name].numel() * clean_state[name].element_size() for name in eligible)})
        write_json(results_dir / "storage.json", storage)

        loaders = make_loaders(sources)
        clean.to(device)
        clustered = build_tinybert(sources.checkpoint["model_config"], decoded).to(device)
        measurements = {}
        for name, loader in loaders.items():
            print(json.dumps({"event": "evaluation_start", "split": name}), flush=True)
            clean_result = evaluate_splits(clean, {name: loader}, device)[name]
            clustered_result = evaluate_splits(clustered, {name: loader}, device)[name]
            measurements[name] = {
                "clean": clean_result, "clustered_no_faults": clustered_result,
                "clustering_accuracy_drop_percentage_points": 100 * (clean_result["accuracy"] - clustered_result["accuracy"]),
                "clustering_cross_entropy_increase": clustered_result["cross_entropy_loss"] - clean_result["cross_entropy_loss"],
                "clean_num_correct_difference_from_clean_eval": clean_result["num_correct"] - clean_metrics["measurements"][name]["num_correct"],
            }
            print(json.dumps({"event": "evaluation_complete", "split": name, **measurements[name]}), flush=True)
        require(all(item["clean_num_correct_difference_from_clean_eval"] == 0 for item in measurements.values()), "Clean replay differs from the two-test clean evaluation")
        require(sha256_file(sources.paths["clean_checkpoint"]) == sources.hashes["clean_checkpoint"], "Clean checkpoint changed during the run")
        write_json(results_dir / "manifest.json", {
            **{key: value for key, value in metadata.items() if key != "model_config"}, **{key: scope[key] for key in ("num_eligible_tensors", "eligible_weights", "total_parameters", "eligible_fraction_of_parameters", "embedding_fraction_of_parameters")},
            "eligible_tensor_names": eligible, "source_artifacts": source_artifacts(root, sources.paths, sources.hashes), "test_set_names": sources.test_names,
            "clustering": {"algorithm": "full-data scalar KMeans / Lloyd (memory/clustering.py, unchanged)", "initialization": "k-means++", "n_init": config.n_init, "max_iter": config.max_iter, "tol": config.tol,
                           "seed_rule": "(seed + tensor ordinal) modulo 2**32", "sampling": "none; fit every eligible scalar weight", "selection": "fixed cluster count; no tuning against any evaluation result"},
            "evaluation": {"splits": list(loaders), "reporting": "each test set separately; never pooled; selection is not a held-out estimate", "training": "none", "batch_size": sources.config.evaluation_batch_size},
        })
        metrics = {"run_id": config.run_id, "num_clusters": config.num_clusters, "faults_injected": 0, "measurements": measurements,
                   "source_checkpoint_sha256": sources.hashes["clean_checkpoint"], "encoding_sha256": sha256_file(encoded_path), "encoding_fingerprint": encoding_fingerprint(saved_encoding),
                   "clustered_dense_sha256": sha256_file(dense_path), "elapsed_seconds": time.monotonic() - started, "completed_at_utc": datetime.now(timezone.utc).isoformat()}
        write_json(results_dir / "metrics.json", metrics)
        write_json(results_dir / "status.json", {"status": "complete", "completed_at_utc": metrics["completed_at_utc"]})
        print(json.dumps({"event": "clustered_baseline_complete", **metrics}), flush=True)
        return metrics
    except BaseException as error:
        write_json(results_dir / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "error_type": type(error).__name__, "error": str(error)})
        raise


def main(project_root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(project_root) / "configs/tinybert_sst2_clustered.json")
    args = parser.parse_args()
    run_tinybert_clustered(TinyBertClusteredConfig(**json.loads(args.config.read_text())), project_root)
