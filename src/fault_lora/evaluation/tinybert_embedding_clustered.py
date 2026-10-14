"""Condition B: cluster TinyBERT's Linear weights and its three embedding tables with Step 4 settings.

Linear weights come first, in module order, so they keep Step 10b's per-tensor
seeds and must reproduce Step 10b's records exactly; the word, position, and
token-type tables follow as ordinals 26–28. LayerNorm parameters and biases
stay reliable untouched state.
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

import torch

from fault_lora.evaluation.clustered_baseline import save_torch, sha256_file, write_json
from fault_lora.evaluation.fault_checks import require, require_state_equal
from fault_lora.evaluation.tinybert_clean_eval import environment_record, source_artifacts
from fault_lora.evaluation.tinybert_clustered import TinyBertClusteredConfig, prepare_clustered_sources
from fault_lora.evaluation.tinybert_fault_checks import load_clustered_run
from fault_lora.evaluation.tinybert_sources import build_tinybert, evaluate_splits, linear_weight_names, make_loaders, protected_state
from fault_lora.memory.clustering import decode_layer, decode_state, physical_levels, storage_estimates
from fault_lora.memory.faults import encoding_fingerprint
from fault_lora.memory.selected_clustering import cluster_named_tensors
from fault_lora.models.tinybert import linear_embedding_layout
from fault_lora.training.baseline import choose_device

REPRESENTATION_B = "per-tensor scalar codebooks and indices for every nn.Linear weight and every nn.Embedding table; no faults or adapters"
CONDITION_B = "linear-and-embeddings"


def embedding_weight_names(model):
    return [layer["weight_name"] for layer in linear_embedding_layout(model) if layer["type"] == "Embedding"]


def condition_b_names(model):
    """Linear weights in module order (Step 10b ordinals), then the embedding tables."""
    return linear_weight_names(model) + embedding_weight_names(model)


@dataclass
class EmbeddingClusteredConfig:
    run_id: str = "tinybert-sst2-clustered-k16-linear-emb-seed42"
    num_clusters: int = 16
    seed: int = 42
    n_init: int = 3
    max_iter: int = 100
    tol: float = 1e-4
    clustering_threads: int = 4
    device: str = "cpu"
    clean_checkpoint_sha256: str = "eace4b3b8cb22e94151b3091b9a6b216afc1443d371cb449f68ba5d512d347f6"
    source_clean_eval_results: str = "results/tinybert-sst2-clean-twotest-seed42"
    step10b_clustered_results: str = "results/tinybert-sst2-clustered-k16-seed42"
    checkpoint_root: str = "checkpoints"
    results_root: str = "results"

    def step10b_style(self):
        """The Step 10b configuration type, used for its validation and its verified source loading."""
        return TinyBertClusteredConfig(run_id=self.run_id, num_clusters=self.num_clusters, seed=self.seed, n_init=self.n_init, max_iter=self.max_iter, tol=self.tol,
                                       clustering_threads=self.clustering_threads, device=self.device, clean_checkpoint_sha256=self.clean_checkpoint_sha256,
                                       source_clean_eval_results=self.source_clean_eval_results, checkpoint_root=self.checkpoint_root, results_root=self.results_root)

    def validate(self):
        self.step10b_style().validate()
        if not isinstance(self.step10b_clustered_results, str) or not self.step10b_clustered_results:
            raise ValueError("step10b_clustered_results is required")


def run_embedding_clustered(config, project_root):
    config.validate()
    root = Path(project_root).resolve()
    device = choose_device(config.device)
    torch.set_num_threads(config.clustering_threads)
    sources, clean_metrics = prepare_clustered_sources(root, config.step10b_style())
    step10b_config, step10b_sources, _, step10b_metrics, step10b_encoding, _ = load_clustered_run(root, config.step10b_clustered_results)
    require(step10b_sources.hashes["clean_checkpoint"] == sources.hashes["clean_checkpoint"], "Step 10b clustered a different clean checkpoint")
    for name in ("num_clusters", "seed", "n_init", "max_iter", "tol"):
        require(getattr(step10b_config, name) == getattr(config, name), f"Clustering setting differs from Step 10b: {name}")
    for name in ("encoding", "clustered_dense", "clustered_metrics", "clustered_status", "clustered_config"):
        sources.paths[f"step10b_{name}"], sources.hashes[f"step10b_{name}"] = step10b_sources.paths[name], step10b_sources.hashes[name]
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
        files = sorted((root / "src/fault_lora").rglob("*.py")) + [root / "scripts/run_tinybert_embedding_clustered.py", root / "requirements.txt", root / "requirements-macos-arm64.lock.txt"]
        write_json(results_dir / "source_sha256.json", {str(path.relative_to(root)): sha256_file(path) for path in files})
        environment = {**environment_record(device, config.clustering_threads, ("scikit-learn", "threadpoolctl")), "clustering_device": "cpu"}
        write_json(results_dir / "environment.json", environment)
        write_json(results_dir / "test_sets.json", sources.test_sets)
        clean = sources.model
        clean_state = {name: value.clone() for name, value in clean.state_dict().items()}
        eligible = condition_b_names(clean)
        protected = protected_state(clean, set(eligible))
        require(protected["embedding_tables"]["num_tensors"] == 0 and protected["other"]["num_tensors"] == 0, "Only LayerNorm and biases may stay reliable")
        scope = {"condition": CONDITION_B, "eligible_tensor_names": eligible, "num_eligible_tensors": len(eligible),
                 "num_linear_tensors": len(linear_weight_names(clean)), "num_embedding_tensors": len(embedding_weight_names(clean)),
                 "eligible_weights": sum(clean_state[name].numel() for name in eligible),
                 "linear_weights": sum(clean_state[name].numel() for name in linear_weight_names(clean)),
                 "embedding_weights": sum(clean_state[name].numel() for name in embedding_weight_names(clean)),
                 "total_parameters": sum(value.numel() for value in clean.parameters()), "protected": protected}
        scope["eligible_fraction_of_parameters"] = scope["eligible_weights"] / scope["total_parameters"]
        write_json(results_dir / "scope.json", scope)
        print(json.dumps({"event": "eligible_tensors", "count": len(eligible), "weights": scope["eligible_weights"]}), flush=True)

        def progress(name, statistics):
            print(json.dumps({"event": "layer_clustered", "tensor": name, **{key: statistics[key] for key in ("num_weights", "actual_clusters", "solver_iterations", "relative_l2_weight_error")}}), flush=True)

        encoding, layers = cluster_named_tensors(clean, eligible, config.num_clusters, seed=config.seed, n_init=config.n_init, max_iter=config.max_iter, tol=config.tol,
                                                 threads=config.clustering_threads, progress=progress)
        require_state_equal(dict(clean.state_dict()), clean_state, "Clean model after clustering")
        # Same tensor, seed, and settings as Step 10b: every Linear record must be identical.
        for name, record in step10b_encoding["clustered_tensors"].items():
            for key in ("codebook", "indices", "index_to_level", "level_to_index"):
                require(torch.equal(encoding["clustered_tensors"][name][key], record[key]), f"Linear record differs from Step 10b: {name}/{key}")
        metadata = {
            "architecture": sources.checkpoint["architecture"], "model_config": sources.checkpoint["model_config"], "num_classes": sources.checkpoint["num_classes"],
            "classes": sources.checkpoint["classes"], "dataset": sources.checkpoint["dataset"], "preprocessing": sources.checkpoint["preprocessing"],
            "split_sha256": sources.checkpoint["split_sha256"], "test_set_sha256": {name: item["sha256"] for name, item in sources.test_sets.items()},
            "source_checkpoint": sources.metrics["checkpoint"], "source_checkpoint_sha256": sources.hashes["clean_checkpoint"],
            "clustering_config": asdict(config), "representation": REPRESENTATION_B, "condition": CONDITION_B,
            "scope": "every nn.Linear weight (Step 10b ordinals 0–25) and the word, position, and token-type embedding tables (ordinals 26–28)",
            "reliable_state": "LayerNorm parameters and all biases are kept exactly and never faulted",
            "physical_mapping": "sorted codebook; identity index-to-level; one cell per index; number of levels equals actual codebook size",
            "codebook_storage_assumption": "reliable codebooks and mapping metadata", "faults_injected": 0, "environment": environment,
        }
        encoding.update(metadata)
        encoded_path = checkpoint_dir / "encoding.pt"
        save_torch(encoded_path, encoding)
        saved_encoding = torch.load(encoded_path, map_location="cpu", weights_only=True)
        decoded = decode_state(saved_encoding)
        for name, record in saved_encoding["clustered_tensors"].items():
            require(torch.equal(decoded[name], decode_layer(record, levels=physical_levels(record))), f"Physical-level and index decoding disagree: {name}")
            layers[name].update({"codebook": record["codebook"].tolist(), "index_to_level": record["index_to_level"].tolist(), "level_to_index": record["level_to_index"].tolist()})
        require(list(saved_encoding["clustered_tensors"]) == eligible, "Saved clustered tensors differ from the eligible list")
        require_state_equal(saved_encoding["untouched_state"], {name: value for name, value in clean_state.items() if name not in eligible}, "Untouched reliable state")
        dense_path = checkpoint_dir / "clustered_dense.pt"
        save_torch(dense_path, {"model_state_dict": decoded, **metadata})
        write_json(results_dir / "layers.json", layers)
        storage = storage_estimates(saved_encoding)
        storage.update({"serialized_encoding_bytes": encoded_path.stat().st_size, "serialized_dense_checkpoint_bytes": dense_path.stat().st_size,
                        "serialized_clean_checkpoint_bytes": sources.paths["clean_checkpoint"].stat().st_size})
        write_json(results_dir / "storage.json", storage)

        loaders = make_loaders(sources)
        clustered = build_tinybert(sources.checkpoint["model_config"], decoded).to(device)
        clean.to(device)
        measurements = {}
        for name, loader in loaders.items():
            clean_result = evaluate_splits(clean, {name: loader}, device)[name]
            clustered_result = evaluate_splits(clustered, {name: loader}, device)[name]
            linear_only = step10b_metrics["measurements"][name]["clustered_no_faults"]
            measurements[name] = {
                "clean": clean_result, "clustered_no_faults": clustered_result,
                "clustering_accuracy_drop_percentage_points": 100 * (clean_result["accuracy"] - clustered_result["accuracy"]),
                "clustering_cross_entropy_increase": clustered_result["cross_entropy_loss"] - clean_result["cross_entropy_loss"],
                "step10b_linear_only_clustered": linear_only,
                "additional_drop_from_embedding_clustering_percentage_points": 100 * (linear_only["accuracy"] - clustered_result["accuracy"]),
                "clean_num_correct_difference_from_clean_eval": clean_result["num_correct"] - clean_metrics["measurements"][name]["num_correct"],
            }
            print(json.dumps({"event": "evaluation_complete", "split": name, "clean": clean_result["accuracy"], "clustered": clustered_result["accuracy"]}), flush=True)
        require(all(item["clean_num_correct_difference_from_clean_eval"] == 0 for item in measurements.values()), "Clean replay differs from the two-test clean evaluation")
        for name, path in sources.paths.items():
            require(sha256_file(path) == sources.hashes[name], f"Source artifact changed: {name}")
        write_json(results_dir / "manifest.json", {
            **{key: value for key, value in metadata.items() if key != "model_config"},
            **{key: scope[key] for key in ("num_eligible_tensors", "eligible_weights", "linear_weights", "embedding_weights", "total_parameters", "eligible_fraction_of_parameters")},
            "eligible_tensor_names": eligible, "source_artifacts": source_artifacts(root, sources.paths, sources.hashes), "test_set_names": sources.test_names,
            "linear_records_identical_to_step10b": True,
            "clustering": {"algorithm": "full-data scalar KMeans / Lloyd via the unchanged clustering.cluster_tensor", "initialization": "k-means++", "n_init": config.n_init,
                           "max_iter": config.max_iter, "tol": config.tol, "seed_rule": "(seed + tensor ordinal) modulo 2**32; Linear ordinals as in Step 10b, embeddings 26–28",
                           "selection": "fixed cluster count; no tuning against any evaluation result"},
            "evaluation": {"splits": list(loaders), "reporting": "each test set separately; never pooled; selection is not a held-out estimate", "training": "none"},
        })
        metrics = {"run_id": config.run_id, "condition": CONDITION_B, "num_clusters": config.num_clusters, "faults_injected": 0, "measurements": measurements,
                   "source_checkpoint_sha256": sources.hashes["clean_checkpoint"], "encoding_sha256": sha256_file(encoded_path), "encoding_fingerprint": encoding_fingerprint(saved_encoding),
                   "clustered_dense_sha256": sha256_file(dense_path), "elapsed_seconds": time.monotonic() - started, "completed_at_utc": datetime.now(timezone.utc).isoformat()}
        write_json(results_dir / "metrics.json", metrics)
        write_json(results_dir / "status.json", {"status": "complete", "completed_at_utc": metrics["completed_at_utc"]})
        print(json.dumps({"event": "embedding_clustered_baseline_complete", "elapsed_seconds": metrics["elapsed_seconds"]}), flush=True)
        return metrics
    except BaseException as error:
        write_json(results_dir / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "error_type": type(error).__name__, "error": str(error)})
        raise


def main(project_root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(project_root) / "configs/tinybert_sst2_clustered_linear_emb.json")
    args = parser.parse_args()
    run_embedding_clustered(EmbeddingClusteredConfig(**json.loads(args.config.read_text())), project_root)
