"""Print per-level fault probabilities for a device fault model; no artifacts are read or written."""

import argparse
from pathlib import Path
import statistics
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fault_lora.memory.device_models import DEVICES, device_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=DEVICES, default="ctt_maxnvm2019")
    parser.add_argument("--levels", type=int, default=16)
    parser.add_argument("--read-noise-ua", type=float, default=0.0, help="CTT only: read-circuit spread added in quadrature")
    parser.add_argument("--l0-gap-ua", type=float, default=None, help="CTT only: gap between L0 and L1")
    parser.add_argument("--sigma-prog-ua", type=float, default=None, help="CTT only: one spread for all programmed levels")
    parser.add_argument("--sigma-mode", choices=("flat", "interpolated"), default="flat")
    args = parser.parse_args()
    model = device_model(args.device, args.levels, read_noise_ua=args.read_noise_ua, l0_gap_ua=args.l0_gap_ua,
                         sigma_prog_ua=args.sigma_prog_ua, sigma_mode=args.sigma_mode)
    means, sigmas = model.parameters.get("level_means_ua"), model.parameters.get("level_sigmas_ua")
    print(model.name)
    print(f"{'level':>5} {'mean uA':>9} {'sigma uA':>9} {'p(down)':>10} {'p(up)':>10} {'p(fault)':>10}")
    for level, (down, up) in enumerate(zip(model.p_down, model.p_up)):
        mean = f"{means[level]:9.2f}" if means else f"{'-':>9}"
        sigma = f"{sigmas[level]:9.2f}" if sigmas else f"{'-':>9}"
        print(f"{level:>5} {mean} {sigma} {down:10.3e} {up:10.3e} {down + up:10.3e}")
    print(f"average over levels used equally: {100 * statistics.mean(model.fault_probability):.3g}%")


if __name__ == "__main__":
    main()
