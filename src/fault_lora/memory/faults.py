"""Saved adjacent-physical-level faults, independent of model/data evaluation.

This uniform, one-cell-per-index model is an experimental simplification, not
a calibrated eNVM model. Applying a pattern always starts from its baseline.
"""

import hashlib
import json
import math
from numbers import Real

import torch

from fault_lora.memory.clustering import decode_layer, decode_state, physical_levels, validate_record


KIND = "static_adjacent_physical_levels_v1"


def validate_parameters(probability, seed):
    if isinstance(probability, bool) or not isinstance(probability, Real) or not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("fault_probability must be finite and between 0 and 1")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")


def encoding_fingerprint(encoding):
    """Hash semantic storage content, independent of torch archive layout."""
    if encoding["format_version"] != 1:
        raise ValueError("Unsupported clustered representation format")
    records, untouched = encoding["clustered_tensors"], encoding["untouched_state"]
    if set(records) & set(untouched):
        raise ValueError("A tensor cannot be both clustered and untouched")
    digest = hashlib.sha256(b"fault_lora.clustered_encoding.v1\n")

    def add_tensor(name, value):
        if value.device.type != "cpu":
            raise ValueError("Saved storage tensors must be on CPU")
        descriptor = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
        digest.update(len(descriptor).to_bytes(8, "little"))
        digest.update(descriptor)
        digest.update(value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())

    for name in sorted(records):
        record = records[name]
        validate_record(record)
        for key in ("codebook", "indices", "index_to_level", "level_to_index"):
            add_tensor(f"clustered/{name}/{key}", record[key])
    for name in sorted(untouched):
        add_tensor(f"untouched/{name}", untouched[name])
    return digest.hexdigest()


def generate_fault_pattern(encoding, fault_probability, *, seed=42, tensor_names=None):
    """Sample Bernoulli-selected cells and persist exact sparse transitions.

    K=1 tensors have no neighboring state and are excluded from the denominator.
    Interior directions are equiprobable; selected boundary cells move inward.
    """
    validate_parameters(fault_probability, seed)
    fingerprint = encoding_fingerprint(encoding)
    records = encoding["clustered_tensors"]
    names = sorted(records if tensor_names is None else tensor_names)
    if not names or len(names) != len(set(names)) or not set(names) <= set(records):
        raise ValueError("Select a nonempty, unique subset of clustered tensor names")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tensors = {}
    for name in names:
        record = records[name]
        k = record["codebook"].numel()
        original = physical_levels(record).reshape(-1)
        positions = torch.empty(0, dtype=torch.int64)
        if k > 1:
            positions = (torch.rand(original.numel(), generator=generator, dtype=torch.float64) < fault_probability).nonzero().flatten()
        before = original[positions].clone()
        steps = torch.randint(0, 2, (positions.numel(),), generator=generator, dtype=torch.int16) * 2 - 1
        steps[before == 0] = 1
        steps[before == k - 1] = -1
        after = (before.to(torch.int16) + steps).to(torch.uint8)
        tensors[name] = {"shape": list(record["shape"]), "num_levels": k,
                         "positions": positions, "before_levels": before, "after_levels": after}
    return {
        "format_version": 1, "kind": KIND, "source_fingerprint": fingerprint,
        "fault_probability": float(fault_probability), "seed": seed,
        "tensor_order": names, "tensors": tensors,
        "sampling": "CPU torch.Generator; sorted tensor names; float64 rand < p per cell, then randint directions per selected cell",
        "torch_version": str(torch.__version__),
        "cell_representation": "one cell per saved index; physical states 0 through K-1; saved inverse mappings used for decoding",
        "interior_up_probability_given_selected": 0.5,
        "boundary_behavior": "selected level 0 moves to 1; selected level K-1 moves to K-2; K=1 tensors excluded",
        "static": True,
        "assumptions": "uniform cell selection; reliable codebooks, mappings, biases, and normalization state; no hardware calibration",
    }


def summarize_fault_pattern(encoding, pattern):
    """Validate exact transitions and compute counts from the saved pattern."""
    if pattern["format_version"] != 1 or pattern["kind"] != KIND:
        raise ValueError("Unsupported fault pattern format")
    validate_parameters(pattern["fault_probability"], pattern["seed"])
    if pattern["source_fingerprint"] != encoding_fingerprint(encoding):
        raise ValueError("Pattern belongs to a different uncorrupted encoding")
    names = pattern["tensor_order"]
    records = encoding["clustered_tensors"]
    if not names or names != sorted(set(names)) or set(names) != set(pattern["tensors"]) or not set(names) <= set(records):
        raise ValueError("Invalid pattern tensor scope/order")
    layers = {}
    totals = {key: 0 for key in ("eligible_cells", "actual_corrupted_cells", "decoded_weight_changes", "upward_transitions", "downward_transitions", "lower_boundary_transitions", "upper_boundary_transitions")}
    for name in names:
        record, entry = records[name], pattern["tensors"][name]
        k = record["codebook"].numel()
        original = physical_levels(record).reshape(-1)
        positions, before, after = entry["positions"], entry["before_levels"], entry["after_levels"]
        if entry["shape"] != record["shape"] or entry["num_levels"] != k:
            raise ValueError(f"Pattern storage layout differs: {name}")
        if any(value.device.type != "cpu" or value.ndim != 1 for value in (positions, before, after)) or positions.dtype != torch.int64 or before.dtype != torch.uint8 or after.dtype != torch.uint8:
            raise ValueError(f"Invalid sparse transition tensors: {name}")
        if not positions.shape == before.shape == after.shape:
            raise ValueError(f"Transition lengths differ: {name}")
        if positions.numel():
            if positions[0].item() < 0 or positions[-1].item() >= original.numel() or not torch.all(positions[1:] > positions[:-1]):
                raise ValueError(f"Positions must be sorted, unique, and in range: {name}")
            if k < 2 or before.max().item() >= k or after.max().item() >= k or not torch.all((after.to(torch.int16) - before.to(torch.int16)).abs() == 1):
                raise ValueError(f"Invalid adjacent-level transition: {name}")
            if not torch.equal(original[positions], before):
                raise ValueError(f"Stored original levels differ from baseline: {name}")
        old_weights = record["codebook"][record["level_to_index"][before.long()]]
        new_weights = record["codebook"][record["level_to_index"][after.long()]]
        layer = {
            "shape": list(record["shape"]), "num_levels": k, "stored_cells": original.numel(),
            "eligible_cells": original.numel() if k > 1 else 0,
            "exclusion_reason": "no adjacent state (K=1)" if k == 1 else None,
            "actual_corrupted_cells": positions.numel(),
            "decoded_weight_changes": int((old_weights != new_weights).sum().item()),
            "upward_transitions": int((after > before).sum().item()),
            "downward_transitions": int((after < before).sum().item()),
            "lower_boundary_transitions": int((before == 0).sum().item()),
            "upper_boundary_transitions": int((before == k - 1).sum().item()),
            "selected_source_level_histogram": torch.bincount(before.long(), minlength=k).tolist(),
            "destination_level_histogram": torch.bincount(after.long(), minlength=k).tolist(),
        }
        layer["realized_fault_rate"] = positions.numel() / layer["eligible_cells"] if layer["eligible_cells"] else 0.0
        layers[name] = layer
        for key in totals:
            totals[key] += layer[key]
    return {**totals, "requested_fault_probability": pattern["fault_probability"],
            "expected_corrupted_cells": totals["eligible_cells"] * pattern["fault_probability"],
            "realized_fault_rate": totals["actual_corrupted_cells"] / totals["eligible_cells"] if totals["eligible_cells"] else 0.0,
            "rate_denominator": "selected clustered tensors with at least two physical levels; one scalar index per cell",
            "layers": layers}


def apply_fault_pattern(encoding, pattern):
    """Return independent dense state from the baseline and saved static reads.

    No input is modified. A pattern cannot be applied to an already-corrupted
    encoding because the source fingerprint must match the original storage.
    """
    summarize_fault_pattern(encoding, pattern)
    state = decode_state(encoding)
    for name, entry in pattern["tensors"].items():
        record = encoding["clustered_tensors"][name]
        levels = physical_levels(record)
        levels.reshape(-1)[entry["positions"]] = entry["after_levels"]
        state[name] = decode_layer(record, levels=levels)
    return state
