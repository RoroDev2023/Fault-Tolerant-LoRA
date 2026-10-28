#!/usr/bin/env python3
"""Run a SCALE-Sim GEMM sweep across systolic-array sizes and model scenarios.

The repository does not yet contain the upstream SCALE-Sim v3 checkout, so this
script keeps the public contract of the simulator run and makes the hardware
sweep explicit without hard-coding a single array size. When `scalesim` is
available in the environment, it executes real GEMM simulations; otherwise it
prints a clear message and exits.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_ARRAY_SIZES = (16, 64, 128)
SCENARIO_LABELS = {
    "baseline": "baseline_resnet",
    "clustered_lora": "clustered_resnet_lora",
    "faulted_lora": "clustered_resnet_lora_faults",
}


def parse_array_sizes(raw_values: str | None) -> tuple[int, ...]:
    if raw_values is None:
        return DEFAULT_ARRAY_SIZES
    values = []
    for part in raw_values.split(","):
        value = int(part.strip())
        if value <= 0:
            raise ValueError(f"Array size must be positive, got {value!r}.")
        values.append(value)
    return tuple(values)


def scenario_rows() -> dict[str, dict[str, str]]:
    return {
        "baseline": {
            "base_workload": "original_gemm.csv",
            "lora_workload": "original_gemm.csv",
            "description": "Dense ResNet baseline",
        },
        "clustered_lora": {
            "base_workload": "original_gemm.csv",
            "lora_workload": "lora_gemm.csv",
            "description": "Clustered ResNet with LoRA adapters",
        },
        "faulted_lora": {
            "base_workload": "original_gemm.csv",
            "lora_workload": "lora_gemm.csv",
            "description": "Clustered ResNet with LoRA plus injected faults",
        },
    }


def read_workload_manifest(manifest_path: Path) -> list[str]:
    if not manifest_path.exists():
        return []
    payload = json.loads(manifest_path.read_text())
    layers = payload.get("layers", [])
    return [layer.get("name") for layer in layers if layer.get("name")]


def report_rows(compute_report: Path) -> list[dict[str, str]]:
    with compute_report.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    normalized = []
    for row in rows:
        result = {
            key.strip() if isinstance(key, str) else key: value
            for key, value in row.items()
        }
        if "Total Cycles (incl. prefetch)" in result:
            result["Prefetch-inclusive cycles"] = result["Total Cycles (incl. prefetch)"].strip()
        for key in ("Total Cycles", "Prefetch-inclusive cycles"):
            if key in result:
                result[key] = result[key].strip()
        normalized.append(result)
    return normalized


def tally(rows: Iterable[dict[str, str]]) -> tuple[int, int]:
    total_cycles = 0
    prefetch_cycles = 0
    for row in rows:
        total_cycles += int(row.get("Total Cycles", 0))
        prefetch_cycles += int(row.get("Prefetch-inclusive cycles", row.get("Total Cycles", 0)))
    return total_cycles, prefetch_cycles


def ensure_scalesim():
    try:
        from scalesim.scale_config import scale_config
    except ImportError as exc:  # pragma: no cover - depends on external simulator install
        raise SystemExit(
            "SCALE-Sim is not installed in this environment. Install the upstream scale-sim-v3 package "
            "and retry, or run with --dry-run to inspect the generated job plan."
        ) from exc
    return scale_config


def validate_config(cfg_path: Path, array_size: int, bandwidth: int) -> None:
    config_type = ensure_scalesim()
    parsed = config_type()
    parsed.read_conf_file(str(cfg_path))
    if not parsed.valid_conf_flag:
        raise ValueError(f"SCALE-Sim rejected the configuration file: {cfg_path}")
    if parsed.get_array_dims() != (array_size, array_size):
        raise ValueError(
            f"Config {cfg_path} has array dimensions {parsed.get_array_dims()}, expected {(array_size, array_size)}"
        )
    if parsed.get_dataflow() != "os":
        raise ValueError(f"Config {cfg_path} uses {parsed.get_dataflow()!r} dataflow; this sweep expects 'os'")
    if parsed.get_bandwidths_as_list() != [bandwidth]:
        raise ValueError(
            f"Config {cfg_path} has bandwidth {parsed.get_bandwidths_as_list()}, expected {[bandwidth]}"
        )


def maybe_execute_simulation(cfg_path: Path, workload_path: Path, out_dir: Path, layout_path: Path) -> Path:
    from scalesim.scale_sim import scalesim

    out_dir.mkdir(parents=True, exist_ok=True)
    sim = scalesim(
        save_disk_space=True,
        verbose=False,
        config=str(cfg_path),
        topology=str(workload_path),
        layout=str(layout_path),
        input_type_gemm=True,
    )
    sim.run_scale(top_path=str(out_dir))
    candidates = sorted(out_dir.rglob("COMPUTE_REPORT.csv"))
    if not candidates:
        raise FileNotFoundError(f"No COMPUTE_REPORT.csv found under {out_dir} after executing {workload_path}.")
    return candidates[0]


def run_single_case(array_size: int, scenario: str, cfg_path: Path, workload_dir: Path, layout_path: Path,
                    out_root: Path, dry_run: bool, bandwidth: int):
    scenario_info = scenario_rows()[scenario]
    output_dir = out_root / f"pe{array_size}" / SCENARIO_LABELS[scenario]
    if dry_run:
        if not cfg_path.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "array_size": array_size,
            "scenario": scenario,
            "config": str(cfg_path),
            "base_workload": scenario_info["base_workload"],
            "lora_workload": scenario_info["lora_workload"],
            "description": scenario_info["description"],
            "status": "dry_run",
            "total_cycles": 0,
            "prefetch_cycles": 0,
        }

    ensure_scalesim()
    validate_config(cfg_path, array_size, bandwidth)
    base_workload = workload_dir / scenario_info["base_workload"]
    lora_workload = workload_dir / scenario_info["lora_workload"]
    if not base_workload.exists():
        raise FileNotFoundError(f"Missing workload {base_workload}")
    if scenario != "baseline" and not lora_workload.exists():
        raise FileNotFoundError(f"Missing workload {lora_workload}")

    base_report = maybe_execute_simulation(cfg_path, base_workload, output_dir / "base", layout_path)
    base_rows = report_rows(base_report)
    total_cycles, prefetch_cycles = tally(base_rows)
    if scenario == "baseline":
        total_lora_cycles = total_lora_prefetch = 0
    else:
        lora_report = maybe_execute_simulation(cfg_path, lora_workload, output_dir / "lora", layout_path)
        lora_rows = report_rows(lora_report)
        total_lora_cycles, total_lora_prefetch = tally(lora_rows)

    result = {
        "array_size": array_size,
        "scenario": scenario,
        "config": str(cfg_path),
        "base_workload": scenario_info["base_workload"],
        "lora_workload": scenario_info["lora_workload"],
        "description": scenario_info["description"],
        "status": "executed",
        "total_cycles": total_cycles + total_lora_cycles,
        "prefetch_cycles": prefetch_cycles + total_lora_prefetch,
    }
    return result


def build_summary(results: list[dict[str, object]]) -> list[dict[str, object]]:
    baseline_cycles_by_array = {
        int(row["array_size"]): int(row["total_cycles"])
        for row in results
        if row["scenario"] == "baseline"
    }
    for row in results:
        baseline_cycles = baseline_cycles_by_array.get(int(row["array_size"]))
        if baseline_cycles is None:
            continue
        if row["scenario"] == "baseline":
            row["overhead_vs_baseline"] = 0.0
        elif baseline_cycles == 0:
            row["overhead_vs_baseline"] = ""
        else:
            row["overhead_vs_baseline"] = round(100.0 * (int(row["total_cycles"]) / baseline_cycles - 1.0), 4)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workloads", type=Path, default=Path("hardware_sim/scalesim/workloads"),
                        help="directory that contains the needed GEMM workloads")
    parser.add_argument("--config", type=Path, default=Path("hardware_sim/scalesim/scale_user_bw.cfg"),
                        help="SCALE-Sim config file")
    parser.add_argument("--layout", type=Path, default=Path("../scale-sim-v3/layouts/conv_nets/test.csv"),
                        help="layout CSV used by SCALE-Sim")
    parser.add_argument("--out", type=Path, default=Path("results/hardware_sim/scalesim"),
                        help="directory for the sweep output")
    parser.add_argument("--array-sizes", type=str, default="16,64,128",
                        help="Comma-separated PE array sizes to simulate, e.g. 16,64,128")
    parser.add_argument("--scenarios", type=str, default="baseline,clustered_lora,faulted_lora",
                        help="Comma-separated scenarios to run")
    parser.add_argument("--bandwidth", type=int, default=10, help="User-mode bandwidth for generated configs")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the planned jobs without executing the simulator")
    parser.add_argument("--test-run", action="store_true",
                        help="run only one array size and one scenario (defaults to 16x16 baseline)")
    args = parser.parse_args()

    workload_dir = args.workloads
    array_sizes = parse_array_sizes(args.array_sizes)
    selected_scenarios = [name.strip() for name in args.scenarios.split(",") if name.strip()]

    if not selected_scenarios:
        parser.error("--scenarios must select at least one scenario")
    if args.test_run:
        array_sizes = (array_sizes[0],)
        selected_scenarios = (selected_scenarios[0],)

    results = []
    for array_size in array_sizes:
        config_path = args.config.with_name(f"scale_{array_size}x{array_size}_user_bw.cfg")
        if not args.dry_run:
            from hardware_sim.scalesim.make_fixed_bw_config import build_config
            build_config(array_size, array_size, args.bandwidth, out_path=config_path)

        for scenario in selected_scenarios:
            if scenario not in scenario_rows():
                raise ValueError(f"Unsupported scenario {scenario!r}. Supported: {', '.join(sorted(scenario_rows()))}")
            result = run_single_case(
                array_size, scenario, config_path, workload_dir, args.layout, args.out, args.dry_run, args.bandwidth
            )
            results.append(result)

    summary = build_summary(results)
    out_csv = args.out / "sweep_summary.csv"
    args.out.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "array_size", "scenario", "description", "status", "base_workload", "lora_workload",
            "total_cycles", "prefetch_cycles", "overhead_vs_baseline", "config",
        ])
        writer.writeheader()
        for row in summary:
            writer.writerow({
                key: row.get(key, "")
                for key in [
                    "array_size", "scenario", "description", "status", "base_workload", "lora_workload",
                    "total_cycles", "prefetch_cycles", "overhead_vs_baseline", "config",
                ]
            })

    print(f"Wrote {out_csv}")
    if args.dry_run and args.test_run:
        print("Single-case test plan only; no simulator execution was performed.")
    elif args.dry_run:
        print("Dry run only; no simulator execution was performed.")
    elif args.test_run:
        print("Completed one-case SCALE-Sim test run.")
    else:
        print("Completed SCALE-Sim sweep for the requested array sizes and scenarios.")


if __name__ == "__main__":
    main()
