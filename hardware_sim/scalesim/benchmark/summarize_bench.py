#!/usr/bin/env python3
"""Split SCALE-Sim cycle totals from run_gemm.py outputs into compute and stall cycles.

Reads the COMPUTE_REPORT.csv files written by run_gemm.py under
<root>/<run>/{base,adapters}/ and prints one row per run, then a per-layer
table for one chosen run. Compute cycles are Total Cycles minus Stall Cycles.
"""
import argparse
import csv
from pathlib import Path


def read_report(directory):
    reports = list(Path(directory).rglob("COMPUTE_REPORT.csv"))
    if len(reports) != 1:
        raise FileNotFoundError(f"Expected one COMPUTE_REPORT.csv under {directory}, found {reports}")
    with reports[0].open(newline="") as f:
        rows = [{k.strip(): v.strip() for k, v in row.items() if k and k.strip()} for row in csv.DictReader(f)]
    return [{"total": int(r["Total Cycles"]), "stall": int(r["Stall Cycles"]),
             "with_prefetch": int(r["Total Cycles (incl. prefetch)"])} for r in rows]


def totals(rows):
    total = sum(r["total"] for r in rows)
    stall = sum(r["stall"] for r in rows)
    return total, total - stall, stall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("results/hardware_sim/scalesim_bench_128"))
    p.add_argument("--workloads", type=Path, default=Path("hardware_sim/scalesim/workloads"))
    p.add_argument("--layers-for", default="calc", help="run whose per-layer table is printed")
    args = p.parse_args()

    runs = sorted(d for d in args.root.iterdir() if (d / "base").is_dir() and (d / "adapters").is_dir())
    print(f"{'run':14s} {'part':9s} {'total':>11s} {'compute':>9s} {'stall':>11s} {'stall %':>8s}")
    for run in runs:
        base, lora = read_report(run / "base"), read_report(run / "adapters")
        for part, rows in (("base", base), ("lora", lora), ("base+lora", base + lora)):
            total, compute, stall = totals(rows)
            print(f"{run.name:14s} {part:9s} {total:>11,} {compute:>9,} {stall:>11,} {100 * stall / total:>7.1f}%")

    with (args.workloads / "original_gemm.csv").open(newline="") as f:
        names = [row[0] for row in list(csv.reader(f))[1:]]
    run = args.root / args.layers_for
    base, lora = read_report(run / "base"), read_report(run / "adapters")
    print(f"\nPer layer, {args.layers_for} (cycles; stall share of each op in parentheses)")
    print(f"{'layer':24s} {'base':>18s} {'lora_A':>18s} {'lora_B':>18s}")
    fmt = lambda r: f"{r['total']:>9,} ({100 * r['stall'] / r['total']:3.0f}%)"
    for j, name in enumerate(names):
        print(f"{name:24s} {fmt(base[j]):>18s} {fmt(lora[2 * j]):>18s} {fmt(lora[2 * j + 1]):>18s}")


if __name__ == "__main__":
    main()
