"""Scalar weight clustering and a fault-free codebook/index representation."""

import math

import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
import torch
from torch import nn


INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def eligible_weight_names(model):
    """Select Conv2d/Linear weights; exclude biases and normalization state."""
    return [f"{name}.weight" if name else "weight" for name, module in model.named_modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))]


def validate_record(record):
    codebook, indices = record["codebook"], record["indices"]
    if codebook.device.type != "cpu" or indices.device.type != "cpu":
        raise ValueError("Stored codebooks and indices must be on CPU")
    if codebook.dtype not in (torch.float32, torch.float64) or codebook.ndim != 1:
        raise ValueError("Codebook must be a float32/float64 vector")
    k = codebook.numel()
    if not 1 <= k <= 256 or not torch.isfinite(codebook).all():
        raise ValueError("Codebook must contain 1–256 finite representatives")
    if not torch.all(codebook[1:] >= codebook[:-1]):
        raise ValueError("Codebook representatives must be sorted")
    if str(codebook.dtype) != record["dtype"]:
        raise ValueError("Recorded dtype differs from codebook dtype")
    if list(indices.shape) != record["shape"] or indices.numel() == 0:
        raise ValueError("Index shape differs from recorded tensor shape or is empty")
    if indices.dtype not in INTEGER_DTYPES or indices.min().item() < 0 or indices.max().item() >= k:
        raise ValueError("Indices must be integers within the codebook")
    expected = torch.arange(k, dtype=torch.int64)
    for key in ("index_to_level", "level_to_index"):
        mapping = record[key]
        if mapping.device.type != "cpu" or mapping.dtype != torch.int64 or mapping.shape != expected.shape:
            raise ValueError("Physical mappings must be CPU int64 vectors of codebook length")
        if not torch.equal(mapping.sort().values, expected):
            raise ValueError("Physical mappings must be permutations of all indices/levels")
    if not torch.equal(record["level_to_index"][record["index_to_level"]], expected):
        raise ValueError("Physical mappings must be inverses")


def physical_levels(record):
    """Encode saved indices into physical level numbers without corruption."""
    validate_record(record)
    return record["index_to_level"][record["indices"].long()].to(torch.uint8)


def decode_layer(record, *, levels=None):
    """Reconstruct weights from saved indices or explicitly supplied levels."""
    validate_record(record)
    indices = record["indices"].long()
    if levels is not None:
        if levels.device.type != "cpu" or levels.dtype not in INTEGER_DTYPES or list(levels.shape) != record["shape"]:
            raise ValueError("Physical levels must be CPU integers with the recorded shape")
        if levels.min().item() < 0 or levels.max().item() >= record["codebook"].numel():
            raise ValueError("Physical level outside the configured cell levels")
        indices = record["level_to_index"][levels.long()]
    return record["codebook"][indices].clone()


def cluster_tensor(tensor, num_clusters, *, seed=42, n_init=3, max_iter=100, tol=1e-4, threads=4):
    if isinstance(num_clusters, bool) or not isinstance(num_clusters, int) or not 1 <= num_clusters <= 256:
        raise ValueError("num_clusters must be an integer from 1 to 256")
    if tensor.dtype not in (torch.float32, torch.float64) or tensor.numel() == 0:
        raise ValueError("Clustering requires a nonempty float32/float64 tensor")
    if not torch.isfinite(tensor).all():
        raise ValueError("Cannot cluster non-finite weights")
    if min(n_init, max_iter, threads) < 1 or not math.isfinite(tol) or tol < 0:
        raise ValueError("Invalid KMeans settings")
    original = tensor.detach().cpu().contiguous().clone()
    values = original.numpy().reshape(-1)
    unique, inverse = np.unique(values, return_inverse=True)
    k = min(num_clusters, len(unique))
    if len(unique) <= num_clusters:
        centers, labels, iterations = unique, inverse, 0
        method = "exact distinct values; no quantization needed"
    else:
        estimator = KMeans(n_clusters=k, init="k-means++", n_init=n_init,
                           max_iter=max_iter, tol=tol, random_state=seed, algorithm="lloyd", copy_x=True)
        with threadpool_limits(limits=threads):
            estimator.fit(values.reshape(-1, 1))
        centers, labels = estimator.cluster_centers_.reshape(-1), estimator.labels_
        iterations, method = int(estimator.n_iter_), "full-data scalar KMeans (Lloyd)"
    order = np.argsort(centers, kind="stable")
    remap = np.empty(k, dtype=np.int64)
    remap[order] = np.arange(k)
    record = {
        "shape": list(original.shape), "dtype": str(original.dtype),
        "codebook": torch.from_numpy(centers[order].copy()).to(original.dtype),
        "indices": torch.from_numpy(remap[labels].astype(np.uint8)).reshape(original.shape),
        "index_to_level": torch.arange(k, dtype=torch.int64),
        "level_to_index": torch.arange(k, dtype=torch.int64),
    }
    decoded = decode_layer(record)
    error = decoded.double() - original.double()
    squared_error = error.square().sum().item()
    norm_squared = original.double().square().sum().item()
    stats = {
        "shape": record["shape"], "dtype": record["dtype"], "num_weights": original.numel(),
        "requested_clusters": num_clusters, "actual_clusters": k, "seed": seed,
        "method": method, "solver_iterations": iterations,
        "iterations_equal_max_iter": iterations == max_iter,
        "cluster_occupancy": torch.bincount(record["indices"].reshape(-1).long(), minlength=k).tolist(),
        "mean_squared_weight_error": squared_error / original.numel(),
        "relative_l2_weight_error": math.sqrt(squared_error / norm_squared) if norm_squared else 0.0,
        "max_absolute_weight_error": error.abs().max().item(),
    }
    return record, stats


def cluster_model(model, num_clusters, *, seed=42, n_init=3, max_iter=100, tol=1e-4, threads=4, progress=None):
    original = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    names = eligible_weight_names(model)
    if not names:
        raise ValueError("Model has no eligible Conv2d/Linear weights")
    records, statistics = {}, {}
    for ordinal, name in enumerate(names):
        layer_seed = (seed + ordinal) % (2**32)
        record, stats = cluster_tensor(original[name], num_clusters, seed=layer_seed,
                                       n_init=n_init, max_iter=max_iter, tol=tol, threads=threads)
        records[name], statistics[name] = record, stats
        if progress is not None:
            progress(name, stats)
    untouched = {name: value for name, value in original.items() if name not in records}
    return {"format_version": 1, "clustered_tensors": records, "untouched_state": untouched}, statistics


def decode_state(encoding):
    if encoding["format_version"] != 1:
        raise ValueError("Unsupported clustered representation format")
    if set(encoding["clustered_tensors"]) & set(encoding["untouched_state"]):
        raise ValueError("A tensor cannot be both clustered and untouched")
    state = {name: value.detach().cpu().clone() for name, value in encoding["untouched_state"].items()}
    for name, record in encoding["clustered_tensors"].items():
        state[name] = decode_layer(record)
    return state


def tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def storage_estimates(encoding):
    original_bytes = 0
    index_bytes = 0
    packed_index_bytes = 0
    codebook_bytes = 0
    mapping_bytes = 0
    for record in encoding["clustered_tensors"].values():
        validate_record(record)
        count, k = record["indices"].numel(), record["codebook"].numel()
        original_bytes += count * record["codebook"].element_size()
        index_bytes += tensor_bytes(record["indices"])
        bits_per_index = (k - 1).bit_length()
        packed_index_bytes += math.ceil(count * bits_per_index / 8)
        codebook_bytes += tensor_bytes(record["codebook"])
        mapping_bytes += tensor_bytes(record["index_to_level"]) + tensor_bytes(record["level_to_index"])
    untouched_bytes = sum(tensor_bytes(value) for value in encoding["untouched_state"].values())
    original_bytes += untouched_bytes
    stored_bytes = index_bytes + codebook_bytes + mapping_bytes + untouched_bytes
    ideal_bytes = packed_index_bytes + codebook_bytes + mapping_bytes + untouched_bytes
    return {
        "original_dense_state_tensor_bytes": original_bytes,
        "stored_index_tensor_bytes": index_bytes,
        "ideal_bitpacked_index_bytes": packed_index_bytes,
        "codebook_tensor_bytes": codebook_bytes,
        "mapping_tensor_bytes": mapping_bytes,
        "untouched_state_tensor_bytes": untouched_bytes,
        "stored_representation_tensor_bytes": stored_bytes,
        "ideal_bitpacked_representation_bytes": ideal_bytes,
        "dense_to_stored_tensor_ratio": original_bytes / stored_bytes,
        "dense_to_ideal_bitpacked_ratio": original_bytes / ideal_bytes,
        "note": "Indices are actually uint8. Bit packing is an estimate, not implemented. Counts include codebooks, both mappings, and untouched state, but exclude archive/metadata overhead. These are logical/software sizes, not hardware area, latency, or energy measurements.",
    }
