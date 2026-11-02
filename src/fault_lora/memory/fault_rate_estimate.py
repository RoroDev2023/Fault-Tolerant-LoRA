"""Step 13a: estimate a cell fault probability from an on-chip memory budget.

Analytic only: nothing is trained, sampled, or rerun. Saved encodings are read
for the weight count and the per-level occupancy of their physical levels.

Device model (Donato et al., DAC 2018, Sections 2.2 and 3.1): each stored level
reads as a Gaussian; a fault is a read into an adjacent level; jumps of two or
more levels are negligible. Here the L = 2^b levels are evenly spaced across a
fixed read window, every level has the same sigma, and reference thresholds lie
halfway between neighbors (assumptions, see ASSUMPTIONS). With spacing
Delta = window / (L - 1), each existing neighbor is read with probability
Q(Delta / (2 sigma)).
"""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

BYTES_PER_MB = 2 ** 20
BITS_PER_MB = 8 * BYTES_PER_MB
BITS_PER_CELL = (1, 2, 3, 4)
MODELS = ("resnet18", "tinybert-linear-emb", "tinybert-linear", "custom")
MEANINGS = ("slc-area", "capacity")

MEANING_TEXT = {
    "capacity": "The budget is total stored bits (budget_mb x 8 x 2^20) at whatever density is chosen; fits when cells x b <= budget bits.",
    "slc-area": "The budget is the area of budget_mb x 8 x 2^20 one-bit cells; a multi-level cell is assumed to take the same area as a one-bit cell (Donato et al. Table 1: CTT-SLC and CTT-MLC both 8 F^2), so fits when cells <= that cell count.",
}
ASSUMPTIONS = [
    "1 MB = 2^20 bytes = 8 x 2^20 bits (or one-bit cells).",
    "Index bits = ceil(log2 k). Required bits = N x index bits; codebooks are reported separately and assumed stored reliably elsewhere.",
    "Cells needed = ceil(required bits / b): indices are bit-packed back to back, so for b that does not divide the index width a cell can straddle two indices.",
    "Levels L = 2^b per cell, evenly spaced across one fixed read window shared by all b (the initial-state gap Delta_init of Donato et al. is set equal to the programmed spacing).",
    "Every level has the same Gaussian read sigma, given as sigma_over_window (sigma divided by the full window).",
    "Reference thresholds sit halfway between adjacent level means, so each existing neighbor is read with probability Q(Delta / (2 sigma)); edge levels have one neighbor.",
    "Two-or-more-level jumps are ignored in the fault probability; their probability Q(3 Delta / (2 sigma)) per direction is reported to show it is negligible.",
    "Level occupancy: the saved encoding's physical levels (tensors in sorted name order, row-major, index bits most significant first) form one bit stream cut into b-bit cells; a cell's level is its b bits read as a natural binary number. For b = index bits this is exactly the per-level histogram.",
    "--level0-safe makes level 0 fault-free (it is never misread); level 1 can still be misread as level 0.",
    "Faults are independent across cells and static; no retention drift, temperature, or write errors are modeled.",
]
PAPER_SOURCE = {
    "citation": "M. Donato, B. Reagen, L. Pentecost, U. Gupta, D. Brooks, G.-Y. Wei. On-Chip Deep Neural Network Storage with Multi-Level eNVM. DAC 2018. doi:10.1145/3195970.3196083",
    "pdf": "https://www.eecs.tufts.edu/~mdonato/assets/papers/dac2018-envm.pdf",
    "used_from_paper": [
        {"item": "Per-level read distribution modeled as a Gaussian", "where": "Section 2.2, page 2 (Figure 1 measurements)"},
        {"item": "A fault is a read into an adjacent level, set by reference thresholds; P_E for a programmed level is the sum of both tails (Eq. 2), for the initial level one tail (Eq. 1)", "where": "Section 3.1, page 3, Eqs. 1-2, Figure 3"},
        {"item": "Multi-level jumps are negligible for the configurations considered", "where": "Section 3.1, page 3"},
        {"item": "2 to 16 levels per CTT cell are reasonable to simulate", "where": "Section 2.2, page 2"},
        {"item": "CTT-SLC and CTT-MLC cell size both 8 F^2 (basis of the slc-area meaning)", "where": "Table 1, page 2 (2 MB NVSim comparison, 22 nm)"},
        {"item": "The initial state can be separated from programmed levels to protect it (basis of --level0-safe)", "where": "Sections 3.1 and 4.3.1, pages 3-4"},
    ],
    "not_in_paper": [
        "No numeric sigma, read window, threshold placement, or per-level error probability is printed in the text or tables.",
        "Figure 1 (page 2) plots five measured CTT levels at V_DD = 0.8 V spanning roughly 18-116 uA with 2-sigma bands; sigma can only be read off the plot, and the pdf panel's normalization is not stated. Not used.",
        "The 3- to 16-level distributions came from SPICE Monte Carlo (Section 2.2, Figure 2); their parameters are not given.",
    ],
}


def q_function(x):
    """Gaussian upper tail Q(x) = P(Z > x)."""
    return 0.5 * math.erfc(x / math.sqrt(2))


def index_bits(clusters):
    if isinstance(clusters, bool) or not isinstance(clusters, int) or clusters < 2:
        raise ValueError("clusters must be an integer >= 2")
    return math.ceil(math.log2(clusters))


def cells_needed(num_weights, clusters, bits_per_cell):
    return math.ceil(num_weights * index_bits(clusters) / bits_per_cell)


def gaussian_level_errors(bits_per_cell, sigma_over_window, *, level0_safe=False):
    """Per-level probabilities of reading one level down/up, and of 2+ level jumps."""
    if not sigma_over_window > 0:
        raise ValueError("sigma_over_window must be positive")
    levels = 2 ** bits_per_cell
    spacing = 1 / (levels - 1)
    one = q_function(spacing / (2 * sigma_over_window))
    two = q_function(3 * spacing / (2 * sigma_over_window))
    down = [0.0 if level == 0 else one for level in range(levels)]
    up = [0.0 if level == levels - 1 else one for level in range(levels)]
    down2 = [0.0 if level < 2 else two for level in range(levels)]
    up2 = [0.0 if level > levels - 3 else two for level in range(levels)]
    if level0_safe:
        down[0] = up[0] = down2[0] = up2[0] = 0.0
    return {"levels": levels, "spacing_over_window": spacing, "down": down, "up": up,
            "fault": [d + u for d, u in zip(down, up)], "two_level_jump": [d + u for d, u in zip(down2, up2)]}


def table_level_errors(table, bits_per_cell, *, level0_safe=False):
    """Per-level errors from an explicit table {"down": [...], "up": [...]} (e.g. from a paper)."""
    levels = 2 ** bits_per_cell
    down, up = [float(v) for v in table["down"]], [float(v) for v in table["up"]]
    if len(down) != levels or len(up) != levels:
        raise ValueError(f"Table for b={bits_per_cell} must list {levels} levels")
    if down[0] != 0 or up[-1] != 0:
        raise ValueError("Level 0 cannot read down and the top level cannot read up")
    if any(not 0 <= d <= 1 or not 0 <= u <= 1 or d + u > 1 for d, u in zip(down, up)):
        raise ValueError("Table probabilities must be in [0, 1] with down + up <= 1 per level")
    if level0_safe:
        up[0] = 0.0
    return {"levels": levels, "spacing_over_window": 1 / (levels - 1), "down": down, "up": up,
            "fault": [d + u for d, u in zip(down, up)], "two_level_jump": None}


def bitstream_occupancy(level_arrays, width, bits_per_cell):
    """Fraction of cells at each level when width-bit levels are packed into b-bit cells."""
    levels = np.concatenate([np.asarray(a, dtype=np.uint8).reshape(-1) for a in level_arrays])
    if levels.size == 0 or levels.max() >= 2 ** width:
        raise ValueError("Levels must be nonempty and fit the index width")
    bits = np.unpackbits(levels[:, None], axis=1)[:, 8 - width:].reshape(-1)
    pad = (-bits.size) % bits_per_cell
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    values = bits.reshape(-1, bits_per_cell).astype(np.int64) @ (1 << np.arange(bits_per_cell - 1, -1, -1))
    counts = np.bincount(values, minlength=2 ** bits_per_cell)
    return (counts / counts.sum()).tolist()


def average(weights, values):
    return float(sum(w * v for w, v in zip(weights, values)))


def load_model_source(model, config, root, *, num_weights=None, clusters=16):
    """Weight count, codebook bits, and per-b occupancy for a model, read from its saved encoding."""
    if model == "custom":
        if isinstance(num_weights, bool) or not isinstance(num_weights, int) or num_weights < 1:
            raise ValueError("--num-weights must be a positive integer for --model custom")
        return {"model": model, "num_weights": num_weights, "codebook_bits": None, "occupancy": None,
                "encoding": None, "note": "Custom weight count; no saved encoding, so only uniform level use is available."}
    if num_weights is not None:
        raise ValueError("--num-weights is only for --model custom")
    import torch
    from fault_lora.memory.clustering import physical_levels
    from fault_lora.memory.faults import encoding_fingerprint

    path = root / config["encodings"][model]
    encoding = torch.load(path, map_location="cpu", weights_only=False)
    records = encoding["clustered_tensors"]
    names = sorted(records)
    ks = {records[name]["codebook"].numel() for name in names}
    if ks != {clusters}:
        raise ValueError(f"{model} encoding has codebook sizes {sorted(ks)}, not --clusters {clusters}")
    arrays = [physical_levels(records[name]).reshape(-1).numpy() for name in names]
    width = index_bits(clusters)
    occupancy = {str(b): bitstream_occupancy(arrays, width, b) for b in BITS_PER_CELL}
    return {"model": model, "num_weights": int(sum(a.size for a in arrays)),
            "codebook_bits": int(sum(8 * records[n]["codebook"].numel() * records[n]["codebook"].element_size() for n in names)),
            "occupancy": occupancy,
            "encoding": {"path": config["encodings"][model], "sha256": file_sha256(path), "fingerprint": encoding_fingerprint(encoding),
                         "num_tensors": len(names), "index_to_level": "identity" if all(
                             records[n]["index_to_level"].tolist() == list(range(clusters)) for n in names) else "custom"},
            "note": "Weight count and level occupancy read from the saved encoding."}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def storage_rows(num_weights, clusters):
    width = index_bits(clusters)
    required = num_weights * width
    rows = []
    for b in BITS_PER_CELL:
        cells = cells_needed(num_weights, clusters, b)
        rows.append({"bits_per_cell": b, "levels": 2 ** b, "cells_needed": cells, "required_bits": required,
                     "stored_bits": cells * b, "cells_per_index": width / b,
                     "min_budget_mb": {"slc-area": cells / BITS_PER_MB, "capacity": cells * b / BITS_PER_MB},
                     "matches_current_simulator": b == width,
                     "runnable_now": b == width,
                     "runnable_note": "one 2^b-level cell per index, as in the current simulator" if b == width else
                     "several cells per index (or cells straddling indices); needs a multi-cell simulator that does not exist yet"})
    return rows


def budget_rows(storage, budget_mb):
    capacity = budget_mb * BITS_PER_MB
    rows = []
    for meaning in MEANINGS:
        for item in storage:
            used = item["cells_needed"] if meaning == "slc-area" else item["stored_bits"]
            rows.append({"budget_mb": budget_mb, "budget_meaning": meaning, "bits_per_cell": item["bits_per_cell"],
                         "budget_units": capacity, "used_units": used, "fits": used <= capacity, "headroom": capacity / used})
        fitting = [row["bits_per_cell"] for row in rows if row["budget_meaning"] == meaning and row["fits"]]
        rows_meaning = [row for row in rows if row["budget_meaning"] == meaning]
        for row in rows_meaning:
            row["fewest_levels_that_fit"] = min(fitting) if fitting else None
    return rows


def fault_rows(source, sigmas, *, level0_safe, table=None):
    """Per-b, per-sigma (or table) per-level and average cell fault probabilities."""
    rows = []
    settings = [("table", None)] if table is not None else [("gaussian", s) for s in sigmas]
    for kind, sigma in settings:
        for b in BITS_PER_CELL:
            if kind == "table":
                entry = table["bits_per_cell"].get(str(b))
                if entry is None:
                    continue
                errors = table_level_errors(entry, b, level0_safe=level0_safe)
            else:
                errors = gaussian_level_errors(b, sigma, level0_safe=level0_safe)
            uniform = [1 / errors["levels"]] * errors["levels"]
            occupancy = None if source["occupancy"] is None else source["occupancy"][str(b)]
            rows.append({"source": kind, "sigma_over_window": sigma, "bits_per_cell": b, **errors,
                         "average_fault_uniform": average(uniform, errors["fault"]),
                         "average_fault_occupancy": None if occupancy is None else average(occupancy, errors["fault"]),
                         "average_two_level_jump_uniform": None if errors["two_level_jump"] is None else average(uniform, errors["two_level_jump"]),
                         "average_two_level_jump_occupancy": None if occupancy is None or errors["two_level_jump"] is None
                         else average(occupancy, errors["two_level_jump"]),
                         "occupancy": occupancy})
    return rows


def sigma_for_target(bits_per_cell, target, weights=None, *, level0_safe=False):
    """sigma_over_window at which the weighted average cell fault probability equals target (bisection in log sigma)."""
    levels = 2 ** bits_per_cell
    weights = weights or [1 / levels] * levels

    def rate(sigma):
        return average(weights, gaussian_level_errors(bits_per_cell, sigma, level0_safe=level0_safe)["fault"])

    low, high = 1e-4, 10.0
    if not rate(low) < target < rate(high):
        return None
    for _ in range(200):
        middle = math.sqrt(low * high)
        low, high = (middle, high) if rate(middle) < target else (low, middle)
    return math.sqrt(low * high)


def build_report(settings, source):
    """Budget-independent parts first, then one block per budget."""
    storage = storage_rows(source["num_weights"], settings["clusters"])
    sigmas = [settings["sigma_over_window"]] if settings["sigma_over_window"] is not None else settings["sigma_grid"]
    faults = fault_rows(source, sigmas, level0_safe=settings["level0_safe"], table=settings.get("level_error_table"))
    budgets = sorted({settings["budget_mb"], *settings["compare_budgets_mb"]})
    return {
        "budget_independent": {
            "model": source["model"], "num_weights": source["num_weights"], "clusters": settings["clusters"],
            "index_bits": index_bits(settings["clusters"]), "required_index_bits": source["num_weights"] * index_bits(settings["clusters"]),
            "required_index_mb": source["num_weights"] * index_bits(settings["clusters"]) / BITS_PER_MB,
            "codebook_bits_reported_separately": source["codebook_bits"], "encoding": source["encoding"], "source_note": source["note"],
            "storage": storage, "fault_probabilities": faults,
            "sigma_for_target_fault_probability": [
                {"bits_per_cell": b, "target": target,
                 "sigma_over_window_uniform": sigma_for_target(b, target, level0_safe=settings["level0_safe"]),
                 "sigma_over_window_occupancy": None if source["occupancy"] is None else
                 sigma_for_target(b, target, source["occupancy"][str(b)], level0_safe=settings["level0_safe"])}
                for b in BITS_PER_CELL for target in settings["target_fault_probabilities"]],
            "sigma_mode": "given" if settings["sigma_over_window"] is not None else (
                "table" if settings.get("level_error_table") else "sensitivity sweep (no sigma given; the paper gives none)"),
            "meanings": MEANING_TEXT, "assumptions": ASSUMPTIONS, "device_source": PAPER_SOURCE,
        },
        "primary_budget_mb": settings["budget_mb"], "primary_budget_meaning": settings["budget_meaning"],
        "budgets": {f"{mb:g}": budget_rows(storage, mb) for mb in budgets},
    }


def write_report(report, results):
    results.mkdir(parents=True)
    (results / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with open(results / "budgets.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, ["budget_mb", "budget_meaning", "bits_per_cell", "levels", "cells_needed", "stored_bits",
                                         "used_units", "budget_units", "fits", "headroom", "fewest_levels_that_fit", "runnable_now"])
        writer.writeheader()
        storage = {item["bits_per_cell"]: item for item in report["budget_independent"]["storage"]}
        for rows in report["budgets"].values():
            for row in rows:
                item = storage[row["bits_per_cell"]]
                writer.writerow({**{k: row[k] for k in ("budget_mb", "budget_meaning", "bits_per_cell", "used_units", "budget_units", "fits",
                                                         "headroom", "fewest_levels_that_fit")},
                                 "levels": item["levels"], "cells_needed": item["cells_needed"], "stored_bits": item["stored_bits"],
                                 "runnable_now": item["runnable_now"]})
    with open(results / "fault_probabilities.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "sigma_over_window", "bits_per_cell", "level", "occupancy", "p_down", "p_up", "p_fault", "p_two_level_jump"])
        for row in report["budget_independent"]["fault_probabilities"]:
            for level in range(row["levels"]):
                writer.writerow([row["source"], row["sigma_over_window"], row["bits_per_cell"], level,
                                 None if row["occupancy"] is None else row["occupancy"][level], row["down"][level], row["up"][level],
                                 row["fault"][level], None if row["two_level_jump"] is None else row["two_level_jump"][level]])
        writer.writerow([])
        writer.writerow(["source", "sigma_over_window", "bits_per_cell", "average_fault_occupancy", "average_fault_uniform",
                         "average_two_level_jump_occupancy", "average_two_level_jump_uniform"])
        for row in report["budget_independent"]["fault_probabilities"]:
            writer.writerow([row["source"], row["sigma_over_window"], row["bits_per_cell"], row["average_fault_occupancy"],
                             row["average_fault_uniform"], row["average_two_level_jump_occupancy"], row["average_two_level_jump_uniform"]])


def fmt_p(value):
    return "n/a" if value is None else ("0" if value == 0 else f"{value:.2e}")


def render(report):
    """Readable terminal tables."""
    info = report["budget_independent"]
    lines = [f"Model {info['model']}: N = {info['num_weights']:,} weights, k = {info['clusters']} -> {info['index_bits']} index bits, "
             f"required {info['required_index_bits']:,} bits = {info['required_index_mb']:.3f} MB (1 MB = 2^20 bytes)."]
    if info["codebook_bits_reported_separately"] is not None:
        lines.append(f"Codebooks, stored separately: {info['codebook_bits_reported_separately']:,} bits "
                     f"({info['codebook_bits_reported_separately'] / 8:,.0f} bytes).")
    lines += ["", "Cells per configuration (* = matches the current one-cell-per-index simulator; others not runnable yet):",
              f"{'b':>2} {'L':>3} {'cells needed':>16} {'min MB slc-area':>16} {'min MB capacity':>16}"]
    for item in info["storage"]:
        lines.append(f"{item['bits_per_cell']:>2} {item['levels']:>3} {item['cells_needed']:>16,} {item['min_budget_mb']['slc-area']:>16.3f} "
                     f"{item['min_budget_mb']['capacity']:>16.3f}{' *' if item['runnable_now'] else ''}")
    lines += ["", "Fit and headroom (budget / used), both meanings side by side:",
              f"{'budget MB':>10} {'b':>2}  {'slc-area':>18}  {'capacity':>18}"]
    for mb, rows in report["budgets"].items():
        by = {(r["budget_meaning"], r["bits_per_cell"]): r for r in rows}
        for b in BITS_PER_CELL:
            cells = [f"{'fits' if by[(m, b)]['fits'] else 'NO':>4} {by[(m, b)]['headroom']:>12.3f}x" for m in MEANINGS]
            lines.append(f"{mb:>10} {b:>2}  {cells[0]:>18}  {cells[1]:>18}")
        fewest = {m: by[(m, 1)]["fewest_levels_that_fit"] for m in MEANINGS}
        lines.append(f"{'':>10}     fewest bits/cell that fit: slc-area {fewest['slc-area']}, capacity {fewest['capacity']}")
    lines += ["", f"Cell fault probability ({info['sigma_mode']}); occupancy = saved encoding's level use, uniform = equal use:",
              f"{'source':>8} {'sigma/W':>8} {'b':>2} {'edge level':>11} {'interior':>10} {'avg occupancy':>14} {'avg uniform':>12} {'2-level jump':>13}"]
    for row in info["fault_probabilities"]:
        lines.append(f"{row['source']:>8} {fmt_p(row['sigma_over_window']) if row['sigma_over_window'] else '-':>8} {row['bits_per_cell']:>2} "
                     f"{fmt_p(row['fault'][-1]):>11} {fmt_p(max(row['fault'])):>10} {fmt_p(row['average_fault_occupancy']):>14} "
                     f"{fmt_p(row['average_fault_uniform']):>12} {fmt_p(row['average_two_level_jump_uniform']):>13}")
    lines += ["", "sigma_over_window giving a target average cell fault probability (occupancy weighting; uniform in brackets):",
              f"{'target':>7} " + " ".join(f"{'b=' + str(b):>22}" for b in BITS_PER_CELL)]
    targets = sorted({row["target"] for row in info["sigma_for_target_fault_probability"]})
    by = {(row["bits_per_cell"], row["target"]): row for row in info["sigma_for_target_fault_probability"]}
    for target in targets:
        cells = []
        for b in BITS_PER_CELL:
            row = by[(b, target)]
            occ, uni = row["sigma_over_window_occupancy"], row["sigma_over_window_uniform"]
            cells.append(f"{'n/a' if occ is None else f'{occ:.4f}'} [{'n/a' if uni is None else f'{uni:.4f}'}]")
        lines.append(f"{target:>7g} " + " ".join(f"{c:>22}" for c in cells))
    return "\n".join(lines)


def resolve_settings(config, args):
    settings = {
        "model": args.model or config["model"], "budget_mb": args.budget_mb if args.budget_mb is not None else config["budget_mb"],
        "budget_meaning": args.budget_meaning or config["budget_meaning"], "clusters": args.clusters or config["clusters"],
        "num_weights": args.num_weights, "sigma_over_window": args.sigma_over_window,
        "sigma_grid": config["sigma_grid"], "target_fault_probabilities": config["target_fault_probabilities"], "compare_budgets_mb": config["compare_budgets_mb"],
        "level0_safe": bool(args.level0_safe or config["level0_safe"]), "level_error_table": None, "level_error_table_path": args.level_error_table,
        "mb_definition": "1 MB = 2^20 bytes = 8,388,608 bits",
    }
    if settings["model"] not in MODELS or settings["budget_meaning"] not in MEANINGS:
        raise ValueError("Unknown --model or --budget-meaning")
    if not settings["budget_mb"] > 0 or any(not mb > 0 for mb in settings["compare_budgets_mb"]):
        raise ValueError("Budgets must be positive")
    if args.level_error_table:
        if args.sigma_over_window is not None:
            raise ValueError("Give either --sigma-over-window or --level-error-table, not both")
        table = json.loads(Path(args.level_error_table).read_text())
        if not table.get("source"):
            raise ValueError("A level error table must name its source")
        settings["level_error_table"] = table
    return settings


def default_run_id(settings):
    name = settings["model"] if settings["model"] != "custom" else f"custom-n{settings['num_weights']}"
    return f"fault-rate-estimate-{name}-{settings['budget_mb']:g}mb"


def main(project_root, argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Step 13a: on-chip memory budget -> cells -> cell fault probability.")
    parser.add_argument("--config", default="configs/fault_rate_estimate.json")
    parser.add_argument("--budget-mb", type=float, help="On-chip budget in MB (1 MB = 2^20 bytes). Default from config: 128.")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--num-weights", type=int, help="Weight count for --model custom.")
    parser.add_argument("--clusters", type=int, help="Codebook size k (default 16).")
    parser.add_argument("--budget-meaning", choices=MEANINGS, help="Primary meaning (default slc-area); both are always reported.")
    parser.add_argument("--sigma-over-window", type=float, help="Read sigma / full read window. Omit for a sensitivity sweep.")
    parser.add_argument("--level0-safe", action="store_true", help="Treat level 0 as fault-free.")
    parser.add_argument("--level-error-table", help="JSON with per-level down/up probabilities per b, and its source.")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    root = Path(project_root)
    config = json.loads((root / args.config).read_text())
    settings = resolve_settings(config, args)
    source = load_model_source(settings["model"], config, root, num_weights=settings["num_weights"], clusters=settings["clusters"])
    report = build_report(settings, source)
    run_id = args.run_id or default_run_id(settings)
    results = root / config.get("results_root", "results") / run_id
    if results.exists():
        raise FileExistsError(f"{results} exists; choose a new --run-id")
    report["run_id"] = run_id
    report["inputs"] = {"cli": vars(args), "config_path": args.config, "config": config,
                        "resolved": {k: v for k, v in settings.items() if k != "level_error_table"},
                        "level_error_table": settings["level_error_table"],
                        "source_sha256": {str(p.relative_to(root)): file_sha256(p) for p in
                                          (root / "src/fault_lora/memory/fault_rate_estimate.py", root / "scripts/estimate_fault_rate.py")}}
    write_report(report, results)
    text = render(report)
    (results / "report.txt").write_text(text + "\n")
    print(text)
    print(f"\nWrote {results.relative_to(root)}/report.json, budgets.csv, fault_probabilities.csv, report.txt")
    return report
