"""Level-dependent static adjacent-level faults and design-time cluster-to-level mappings (Step 13b).

Sibling of fault_lora.memory.faults. Patterns use the same sparse layout
(positions, before_levels, after_levels) with a new kind. Each cell is
misread one level down with probability p_down[level] and one level up with
p_up[level]; one float64 uniform u per cell decides both: down if
u < p_down, up if p_down <= u < p_down + p_up. Applying a pattern reuses the
existing decoding (decode_state / decode_layer through level_to_index).
"""

import copy
import math

import torch

from fault_lora.memory.clustering import decode_layer, decode_state, physical_levels, validate_record
from fault_lora.memory.fault_rate_estimate import gaussian_level_errors
from fault_lora.memory.faults import encoding_fingerprint, validate_parameters


KIND = "static_level_dependent_adjacent_levels_v1"
MAPPINGS = ("sorted", "frequent-at-0")


def gaussian_level_probabilities(sigma_over_window, num_levels=16, *, level0_safe=False):
    """Per-level down/up read-error probabilities from the Step 13a Gaussian model."""
    bits = round(math.log2(num_levels))
    if 2 ** bits != num_levels:
        raise ValueError("num_levels must be a power of two")
    errors = gaussian_level_errors(bits, sigma_over_window, level0_safe=level0_safe)
    return {"down": errors["down"], "up": errors["up"], "model": "gaussian", "sigma_over_window": sigma_over_window,
            "level0_safe": level0_safe, "num_levels": num_levels}


def validate_level_probabilities(down, up):
    if len(down) != len(up) or len(down) < 2:
        raise ValueError("down and up must list the same number (>= 2) of levels")
    if down[0] != 0 or up[-1] != 0:
        raise ValueError("Level 0 cannot read down and the top level cannot read up")
    for d, u in zip(down, up):
        if not (isinstance(d, float) and isinstance(u, float)) or not (0 <= d <= 1 and 0 <= u <= 1 and d + u <= 1):
            raise ValueError("Per-level probabilities must be floats in [0, 1] with down + up <= 1")


def generate_level_fault_pattern(encoding, probabilities, *, seed, tensor_names=None):
    """Sample one static pattern; every clustered tensor must have len(down) levels."""
    down, up = [float(v) for v in probabilities["down"]], [float(v) for v in probabilities["up"]]
    validate_level_probabilities(down, up)
    validate_parameters(0, seed)
    fingerprint = encoding_fingerprint(encoding)
    records = encoding["clustered_tensors"]
    names = sorted(records if tensor_names is None else tensor_names)
    if not names or len(names) != len(set(names)) or not set(names) <= set(records):
        raise ValueError("Select a nonempty, unique subset of clustered tensor names")
    p_down, p_up = torch.tensor(down, dtype=torch.float64), torch.tensor(up, dtype=torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tensors = {}
    for name in names:
        record = records[name]
        if record["codebook"].numel() != len(down):
            raise ValueError(f"{name} has {record['codebook'].numel()} levels, probabilities list {len(down)}")
        original = physical_levels(record).reshape(-1)
        draws = torch.rand(original.numel(), generator=generator, dtype=torch.float64)
        level_down, level_up = p_down[original.long()], p_up[original.long()]
        moves_down = draws < level_down
        moves_up = (draws >= level_down) & (draws < level_down + level_up)
        positions = (moves_down | moves_up).nonzero().flatten()
        before = original[positions].clone()
        after = (before.to(torch.int16) + torch.where(moves_down[positions], -1, 1).to(torch.int16)).to(torch.uint8)
        tensors[name] = {"shape": list(record["shape"]), "num_levels": len(down), "positions": positions, "before_levels": before, "after_levels": after}
    return {
        "format_version": 1, "kind": KIND, "source_fingerprint": fingerprint, "seed": seed,
        "level_down_probabilities": down, "level_up_probabilities": up,
        "probability_model": {key: value for key, value in probabilities.items() if key not in ("down", "up")},
        "tensor_order": names, "tensors": tensors,
        "sampling": "CPU torch.Generator; sorted tensor names; one float64 rand per cell: down if u < p_down[level], up if p_down <= u < p_down + p_up",
        "torch_version": str(torch.__version__),
        "cell_representation": "one cell per saved index; physical states 0 through K-1; saved inverse mappings used for decoding",
        "static": True,
        "assumptions": "independent cells; per-level adjacent-read probabilities from the given model; reliable codebooks, mappings, biases, and normalization state",
    }


def summarize_level_fault_pattern(encoding, pattern):
    """Validate the saved transitions; counts per tensor and per source level."""
    if pattern.get("format_version") != 1 or pattern.get("kind") != KIND:
        raise ValueError("Unsupported level fault pattern format")
    down, up = pattern["level_down_probabilities"], pattern["level_up_probabilities"]
    validate_level_probabilities(down, up)
    if pattern["source_fingerprint"] != encoding_fingerprint(encoding):
        raise ValueError("Pattern belongs to a different uncorrupted encoding")
    names = pattern["tensor_order"]
    records = encoding["clustered_tensors"]
    if not names or names != sorted(set(names)) or set(names) != set(pattern["tensors"]) or not set(names) <= set(records):
        raise ValueError("Invalid pattern tensor scope/order")
    k = len(down)
    level_cells, level_down, level_up = torch.zeros(k, dtype=torch.long), torch.zeros(k, dtype=torch.long), torch.zeros(k, dtype=torch.long)
    totals = {"eligible_cells": 0, "actual_corrupted_cells": 0, "decoded_weight_changes": 0, "upward_transitions": 0, "downward_transitions": 0}
    layers = {}
    for name in names:
        record, entry = records[name], pattern["tensors"][name]
        validate_record(record)
        original = physical_levels(record).reshape(-1)
        positions, before, after = entry["positions"], entry["before_levels"], entry["after_levels"]
        if entry["shape"] != record["shape"] or entry["num_levels"] != k or record["codebook"].numel() != k:
            raise ValueError(f"Pattern storage layout differs: {name}")
        if any(value.device.type != "cpu" or value.ndim != 1 for value in (positions, before, after)) or positions.dtype != torch.int64 \
                or before.dtype != torch.uint8 or after.dtype != torch.uint8 or not positions.shape == before.shape == after.shape:
            raise ValueError(f"Invalid sparse transition tensors: {name}")
        if positions.numel():
            if positions[0].item() < 0 or positions[-1].item() >= original.numel() or not torch.all(positions[1:] > positions[:-1]):
                raise ValueError(f"Positions must be sorted, unique, and in range: {name}")
            if before.max().item() >= k or after.max().item() >= k or not torch.all((after.to(torch.int16) - before.to(torch.int16)).abs() == 1):
                raise ValueError(f"Invalid adjacent-level transition: {name}")
            if not torch.equal(original[positions], before):
                raise ValueError(f"Stored original levels differ from baseline: {name}")
        downs, ups = before[after < before].long(), before[after > before].long()
        if any(down[level] == 0 for level in downs.unique().tolist()) or any(up[level] == 0 for level in ups.unique().tolist()):
            raise ValueError(f"A transition has zero requested probability: {name}")
        old_weights = record["codebook"][record["level_to_index"][before.long()]]
        new_weights = record["codebook"][record["level_to_index"][after.long()]]
        level_cells += torch.bincount(original.long(), minlength=k)
        level_down += torch.bincount(downs, minlength=k)
        level_up += torch.bincount(ups, minlength=k)
        layer = {"eligible_cells": original.numel(), "actual_corrupted_cells": positions.numel(),
                 "decoded_weight_changes": int((old_weights != new_weights).sum().item()),
                 "upward_transitions": ups.numel(), "downward_transitions": downs.numel()}
        layer["realized_fault_rate"] = layer["actual_corrupted_cells"] / layer["eligible_cells"]
        layers[name] = layer
        for key in totals:
            totals[key] += layer[key]
    cells = level_cells.tolist()
    expected = sum(c * (d + u) for c, d, u in zip(cells, down, up))
    return {**totals, "expected_corrupted_cells": expected,
            "requested_average_fault_probability": expected / totals["eligible_cells"],
            "realized_fault_rate": totals["actual_corrupted_cells"] / totals["eligible_cells"],
            "per_level": {"cells": cells, "down_transitions": level_down.tolist(), "up_transitions": level_up.tolist(),
                          "requested_down": down, "requested_up": up},
            "rate_denominator": "selected clustered tensors; one scalar index per cell", "layers": layers}


def apply_level_fault_pattern(encoding, pattern):
    """Independent dense state read through the saved static pattern; inputs are not modified."""
    summarize_level_fault_pattern(encoding, pattern)
    state = decode_state(encoding)
    for name, entry in pattern["tensors"].items():
        record = encoding["clustered_tensors"][name]
        levels = physical_levels(record)
        levels.reshape(-1)[entry["positions"]] = entry["after_levels"]
        state[name] = decode_layer(record, levels=levels)
    return state


def per_level_z_scores(summary):
    """Binomial z-score of realized vs requested transitions, per level and direction (None when n p (1 - p) = 0)."""
    rows = []
    level = summary["per_level"]
    for index, cells in enumerate(level["cells"]):
        for direction, key in (("down", "down_transitions"), ("up", "up_transitions")):
            p, count = level[f"requested_{direction}"][index], level[key][index]
            variance = cells * p * (1 - p)
            rows.append({"level": index, "direction": direction, "cells": cells, "requested": p, "count": count, "expected": cells * p,
                         "realized": count / cells if cells else None, "z": (count - cells * p) / math.sqrt(variance) if variance > 0 else None,
                         "impossible_event_occurred": variance == 0 and count != cells * p})
    return rows


def frequent_at_zero_mapping(record):
    """Most frequent index on level 0, the other indices in ascending (sorted-codebook) order on levels 1..K-1."""
    validate_record(record)
    k = record["codebook"].numel()
    counts = torch.bincount(record["indices"].reshape(-1).long(), minlength=k)
    frequent = int(torch.argmax(counts).item())  # ties: lowest index
    order = [frequent] + [index for index in range(k) if index != frequent]
    level_to_index = torch.tensor(order, dtype=torch.int64)
    index_to_level = torch.empty(k, dtype=torch.int64)
    index_to_level[level_to_index] = torch.arange(k, dtype=torch.int64)
    return index_to_level, level_to_index, frequent, counts


def remap_encoding(encoding, mapping):
    """Copy of the encoding with a design-time index-to-level mapping; indices and codebooks unchanged."""
    if mapping not in MAPPINGS:
        raise ValueError(f"mapping must be one of {MAPPINGS}")
    remapped = copy.copy(encoding)
    remapped["clustered_tensors"] = {}
    for name, record in encoding["clustered_tensors"].items():
        k = record["codebook"].numel()
        if mapping == "sorted":
            index_to_level = level_to_index = torch.arange(k, dtype=torch.int64)
            level_to_index = level_to_index.clone()
        else:
            index_to_level, level_to_index, _, _ = frequent_at_zero_mapping(record)
        remapped["clustered_tensors"][name] = {**record, "index_to_level": index_to_level, "level_to_index": level_to_index}
        validate_record(remapped["clustered_tensors"][name])
    remapped["physical_mapping"] = {"sorted": "identity: level = index of the sorted codebook (level 0 = most negative cluster)",
                                    "frequent-at-0": "per tensor: most frequent cluster on level 0, remaining clusters in sorted order on levels 1-15"}[mapping]
    return remapped


def mapping_report(encoding, remapped):
    """Per tensor: share of cells on level 0 under each mapping and the level-1-to-0 jump vs sorted neighbor gaps."""
    rows = []
    for name in sorted(encoding["clustered_tensors"]):
        record, other = encoding["clustered_tensors"][name], remapped["clustered_tensors"][name]
        codebook = record["codebook"].double()
        counts = torch.bincount(record["indices"].reshape(-1).long(), minlength=codebook.numel())
        total = counts.sum().item()
        gaps = codebook[1:] - codebook[:-1]
        frequent = int(other["level_to_index"][0].item())
        level1_index = int(other["level_to_index"][1].item())
        rows.append({"tensor": name, "cells": total, "frequent_index": frequent, "frequent_value": codebook[frequent].item(),
                     "sorted_level0_share": counts[int(record["level_to_index"][0])].item() / total,
                     "frequent_level0_share": counts[frequent].item() / total,
                     "level1_index_under_frequent": level1_index, "level1_share_under_frequent": counts[level1_index].item() / total,
                     "jump_level1_to_0_frequent": abs(codebook[frequent] - codebook[level1_index]).item(),
                     "sorted_median_neighbor_gap": gaps.median().item(), "sorted_mean_neighbor_gap": gaps.mean().item(),
                     "sorted_level1_to_0_gap": (codebook[1] - codebook[0]).item(),
                     "jump_over_median_gap": abs(codebook[frequent] - codebook[level1_index]).item() / gaps.median().item(),
                     "frequent_levels_from_sorted_edge": frequent})
    return rows
