"""Cluster an explicit, ordered list of weight tensors with the unchanged per-tensor algorithm.

`clustering.cluster_model` fixes its eligibility rule (Conv2d/Linear weights).
This wrapper takes the tensor list instead, calls the same `cluster_tensor`
with the same `(seed + ordinal)` rule, and returns the same format-1 encoding,
so decoding, storage accounting, and the fault simulator apply unchanged.
"""

from fault_lora.memory.clustering import cluster_tensor


def cluster_named_tensors(model, names, num_clusters, *, seed=42, n_init=3, max_iter=100, tol=1e-4, threads=4, progress=None):
    """Ordinal i in `names` uses seed (seed + i) modulo 2**32, exactly as `cluster_model` does."""
    original = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    names = list(names)
    if not names or len(set(names)) != len(names):
        raise ValueError("Select a nonempty list of unique tensor names")
    missing = [name for name in names if name not in original]
    if missing:
        raise ValueError(f"Unknown state tensors: {missing}")
    records, statistics = {}, {}
    for ordinal, name in enumerate(names):
        layer_seed = (seed + ordinal) % (2**32)
        record, stats = cluster_tensor(original[name], num_clusters, seed=layer_seed, n_init=n_init, max_iter=max_iter, tol=tol, threads=threads)
        records[name], statistics[name] = record, stats
        if progress is not None:
            progress(name, stats)
    untouched = {name: value for name, value in original.items() if name not in records}
    return {"format_version": 1, "clustered_tensors": records, "untouched_state": untouched}, statistics
