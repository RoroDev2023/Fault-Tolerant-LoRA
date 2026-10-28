#!/usr/bin/env python3
"""Create a SCALE-Sim config with a fixed user-specified bandwidth and PE array size."""

import argparse
from pathlib import Path


def build_config(pe_x: int, pe_y: int, bandwidth: int, *, out_path: Path) -> Path:
    """Write a generic SCALE-Sim config with explicit PE dimensions and bandwidth."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = f"""[global]
InterfaceBandwidth: USER
Dataflow: OS
PE_ARRAY_X: {pe_x}
PE_ARRAY_Y: {pe_y}
Bandwidth: {bandwidth}
IFMAP_SRAM: 128
FILTER_SRAM: 128
OFMAP_SRAM: 128

[sim]
verbose: False
save_disk_space: True
"""
    out_path.write_text(cfg)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pe-x", type=int, default=16, help="PE array width")
    parser.add_argument("--pe-y", type=int, default=16, help="PE array height")
    parser.add_argument("--bandwidth", type=int, default=10, help="Interface bandwidth in SCALE-Sim units")
    parser.add_argument("--out", type=Path, default=Path("hardware_sim/scalesim/scale_user_bw.cfg"),
                        help="output config path")
    args = parser.parse_args()

    config_path = build_config(args.pe_x, args.pe_y, args.bandwidth, out_path=args.out)
    print(f"Wrote {config_path}")


if __name__ == "__main__":
    main()
