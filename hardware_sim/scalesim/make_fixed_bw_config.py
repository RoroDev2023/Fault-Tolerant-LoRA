#!/usr/bin/env python3
"""Generate a SCALE-Sim v3 architecture config with explicit array and bandwidth."""

import argparse
from pathlib import Path
import re


def build_config(
    pe_x: int,
    pe_y: int,
    bandwidth: int,
    *,
    out_path: Path,
    dataflow: str = "os",
    ifmap_sram_kb: int = 128,
    filter_sram_kb: int = 128,
    ofmap_sram_kb: int = 128,
) -> Path:
    """Write a config using the required SCALE-Sim v3 configparser schema."""
    for name, value in (
        ("pe_x", pe_x),
        ("pe_y", pe_y),
        ("bandwidth", bandwidth),
        ("ifmap_sram_kb", ifmap_sram_kb),
        ("filter_sram_kb", filter_sram_kb),
        ("ofmap_sram_kb", ofmap_sram_kb),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if dataflow not in ("os", "ws", "is"):
        raise ValueError("dataflow must be one of: os, ws, is")

    run_name = f"fault_lora_{pe_y}x{pe_x}_{dataflow}_user_bw"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
        raise ValueError(f"Invalid SCALE-Sim run name: {run_name!r}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = f"""[general]
run_name = {run_name}

[architecture_presets]
ArrayHeight: {pe_y}
ArrayWidth: {pe_x}
IfmapSramSzkB: {ifmap_sram_kb}
FilterSramSzkB: {filter_sram_kb}
OfmapSramSzkB: {ofmap_sram_kb}
IfmapOffset: 0
FilterOffset: 10000000
OfmapOffset: 20000000
Bandwidth: {bandwidth}
Dataflow: {dataflow}
ReadRequestBuffer: 32
WriteRequestBuffer: 32

[layout]
IfmapCustomLayout: False
IfmapSRAMBankBandwidth: 10
IfmapSRAMBankNum: 10
IfmapSRAMBankPort: 2
FilterCustomLayout: False
FilterSRAMBankBandwidth: 10
FilterSRAMBankNum: 10
FilterSRAMBankPort: 2

[sparsity]
SparsitySupport: False
SparseRep: ellpack_block
OptimizedMapping: False
BlockSize: 8
RandomNumberGeneratorSeed: 40

[run_presets]
InterfaceBandwidth: USER
UseRamulatorTrace: False
SRAM_row_size: 2
SRAM_bank_size: 5
DRAM_row_size: 2
DRAM_bank_size: 5
"""
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    temporary.write_text(cfg)
    temporary.replace(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pe-x", type=int, default=16, help="array width in PEs")
    parser.add_argument("--pe-y", type=int, default=16, help="array height in PEs")
    parser.add_argument("--bandwidth", type=int, default=10, help="SCALE-Sim user bandwidth")
    parser.add_argument("--dataflow", choices=("os", "ws", "is"), default="os")
    parser.add_argument("--ifmap-sram-kb", type=int, default=128)
    parser.add_argument("--filter-sram-kb", type=int, default=128)
    parser.add_argument("--ofmap-sram-kb", type=int, default=128)
    parser.add_argument("--out", type=Path, default=Path("hardware_sim/scalesim/scale_16x16_user_bw.cfg"),
                        help="output config path")
    args = parser.parse_args()

    try:
        config_path = build_config(
            args.pe_x,
            args.pe_y,
            args.bandwidth,
            out_path=args.out,
            dataflow=args.dataflow,
            ifmap_sram_kb=args.ifmap_sram_kb,
            filter_sram_kb=args.filter_sram_kb,
            ofmap_sram_kb=args.ofmap_sram_kb,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Wrote {config_path}")


if __name__ == "__main__":
    main()
