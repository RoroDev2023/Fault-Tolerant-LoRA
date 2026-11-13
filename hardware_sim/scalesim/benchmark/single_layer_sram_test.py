#!/usr/bin/env python3
"""Run one GEMM layer on SCALE-Sim with different filter SRAM sizes and bandwidths.

Purpose: check whether the large stall counts in the full ResNet-18 runs come from
SCALE-Sim fetching the same weights from DRAM many times when a layer's weights do
not fit in the filter SRAM. For each variant the script reports total and stall
cycles and how many times the layer's weights were read from DRAM
(DRAM Filter Reads / (N * K)).

Example:
  PYTHONPATH=../scale-sim-v3 python hardware_sim/scalesim/benchmark/single_layer_sram_test.py \
      --layout ../scale-sim-v3/layouts/conv_nets/test.csv
"""
import argparse
import configparser
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_csv_rows(directory, name):
    reports = list(Path(directory).rglob(name))
    if len(reports) != 1:
        raise FileNotFoundError(f"Expected one {name} under {directory}, found {reports}")
    with reports[0].open(newline="") as f:
        return [{k.strip(): v.strip() for k, v in row.items() if k and k.strip()} for row in csv.DictReader(f)]


def find_layer(workloads, layer):
    with (Path(workloads) / "original_gemm.csv").open(newline="") as f:
        for row in list(csv.reader(f))[1:]:
            if row and row[0].strip() == layer:
                return [int(v) for v in row[1:4]]
    raise KeyError(f"{layer} not found in {workloads}/original_gemm.csv")


def write_config(base, path, run_name, filter_kb, bandwidth):
    c = configparser.ConfigParser()
    c.read(base)
    c["general"]["run_name"] = run_name
    c["architecture_presets"]["filtersramszkb"] = str(filter_kb)
    if bandwidth is None:
        c["run_presets"]["interfacebandwidth"] = "CALC"
    else:
        # In USER mode SCALE-Sim takes ifmap/filter DRAM bandwidth from the bank
        # bandwidth fields; [architecture_presets] bandwidth only sets ofmap bandwidth.
        c["run_presets"]["interfacebandwidth"] = "USER"
        c["architecture_presets"]["bandwidth"] = str(bandwidth)
        c["layout"]["ifmapsrambankbandwidth"] = str(bandwidth)
        c["layout"]["filtersrambankbandwidth"] = str(bandwidth)
    with path.open("w") as f:
        c.write(f)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layer", default="layer4.0.conv2")
    p.add_argument("--workloads", default="hardware_sim/scalesim/workloads")
    p.add_argument("--base-config", default=str(HERE / "scale_128x128_user_bank160.cfg"))
    p.add_argument("--layout", default="../scale-sim-v3/layouts/conv_nets/test.csv")
    p.add_argument("--out", default="results/hardware_sim/scalesim_single_layer")
    p.add_argument("--filter-kb", type=int, nargs="+", default=[64, 4096],
                   help="filter SRAM sizes to try (KB; SCALE-Sim words are 1 byte)")
    p.add_argument("--bandwidth", type=int, nargs="+", default=[160],
                   help="USER-mode DRAM words/cycle; must be divisible by the bank count (10)")
    p.add_argument("--no-calc", action="store_true", help="skip the CALC (stall-free) reference run")
    args = p.parse_args()

    from scalesim.scale_sim import scalesim

    m, n, k = find_layer(args.workloads, args.layer)
    weights = n * k
    print(f"{args.layer}: M={m} N={n} K={k}; weights = {weights:,} words = {weights / 1024:,.0f} KB")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    topo = out / "layer.csv"
    topo.write_text(f"Layer Name, M, N, K,\n{args.layer},{m},{n},{k},\n")

    variants = [] if args.no_calc else [("calc", None, min(args.filter_kb))]
    variants += [(f"bw{bw}_filter{kb}kb", bw, kb) for bw in args.bandwidth for kb in args.filter_kb]
    rows = []
    for name, bw, kb in variants:
        run_dir = out / name
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = run_dir / "config.cfg"
        write_config(args.base_config, cfg, name, kb, bw)
        sim = scalesim(save_disk_space=True, verbose=False, config=str(cfg), topology=str(topo),
                       layout=str(args.layout), input_type_gemm=True)
        sim.run_scale(top_path=str(run_dir))
        comp = read_csv_rows(run_dir, "COMPUTE_REPORT.csv")[0]
        acc = read_csv_rows(run_dir, "DETAILED_ACCESS_REPORT.csv")[0]
        total, stall = int(comp["Total Cycles"]), int(comp["Stall Cycles"])
        reads = int(float(acc["DRAM Filter Reads"]))
        rows.append({"variant": name, "dram_words_per_cycle": bw or "CALC", "filter_sram_kb": kb,
                     "weights_fit_in_sram": kb * 1024 >= weights, "total_cycles": total, "stall_cycles": stall,
                     "compute_cycles": total - stall, "dram_filter_reads": reads,
                     "weight_fetches": round(reads / weights, 2)})
        print(f"{name:22s} total={total:>11,} stall={stall:>11,} dram_filter_reads={reads:>12,} "
              f"({reads / weights:.1f}x the weights)", flush=True)

    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved", out / "summary.csv")


if __name__ == "__main__":
    main()
