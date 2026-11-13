"""Per-level misread probabilities for multi-level memory cells.

A cell stores one physical level 0..K-1. Its read signal (a current for CTT)
differs from cell to cell, so cells written to the same level form a
distribution. The read circuit compares the signal against one threshold
between each pair of neighbouring levels. A cell whose signal lands past a
threshold is read as the neighbouring level, every time it is read: a static
fault. This module turns per-level signal distributions into a
``LevelFaultModel``: for every physical level, the chance a cell stored there is
read one level down (``p_down``) or one level up (``p_up``).

Physical level i is the technology's level Li. For CTT, L0 is the unprogrammed
state (highest current); each later level holds more trapped charge and less
current. Which weight cluster sits at which level is set by the encoding's
``index_to_level`` mapping, not here.

Data sources (all values are per-level means and spreads of the read signal):

* CTT, ``maxnvm2019``: 8-level cells on a 16 nm FinFET test chip, read current
  at 0.8 V, 128 cells per level. Read from the vector data of Fig. 2b in
  Pentecost, Donato, Reagen et al., "MaxNVM", MICRO 2019; each cell's current
  is its IV curve at 0.8 V. Matches the paper's histogram panel within 0.3 uA.
* CTT, ``dac2018``: 5-level cells, same process, 0.8 V. Read from Fig. 1 of
  Donato et al., "On-Chip Deep Neural Network Storage with Multi-Level eNVM",
  DAC 2018.
* RRAM and FeFET: ``mlc_rram_args.p`` and ``mlc_fefet_args.p`` from
  MemSysExplorer's msxFI (github.com/TuftsECS/MemSysExplorer, tech/msxFI/mem_data),
  2, 4 and 8 levels. RRAM is Gaussian in normalized conductance; FeFET is gamma
  in amperes. Threshold rules follow msxFI.

Only the cell spread is modeled. Whether the read circuit (current-to-voltage
converter, sense amplifier) adds spread is unknown for the msxFI files and not
included in the chip data; ``read_noise_ua`` adds it explicitly for CTT.
"""

from dataclasses import dataclass, field
import math

import torch


# (mean uA, sigma uA) per level, L0 first.
CTT_CHIPS = {
    "maxnvm2019": ((144.70, 5.80), (94.39, 1.10), (79.37, 1.25), (64.21, 1.39),
                   (49.73, 1.40), (34.44, 1.43), (19.66, 1.19), (4.91, 0.57)),
    "dac2018": ((116.30, 4.41), (78.85, 1.13), (58.52, 1.13), (38.44, 1.45), (18.79, 1.09)),
}
# Programmed levels below this current sit near the transistor's turn-on point,
# where the spread shrinks (MaxNVM L7: 0.57 uA). They are left out of the
# default flat programmed-level sigma.
NEAR_TURN_ON_UA = 10.0

# msxFI RRAM: (mean, sigma) per level, normalized conductance, lowest first.
RRAM_MSXFI = {
    2: ((0.0, 0.00864567), (1.0, 0.00339958)),
    4: ((0.0, 0.00864567), (0.38464387, 0.03351085), (0.72165694, 0.02741654), (1.0, 0.00339958)),
    8: ((0.0, 0.00864567), (0.07826194, 0.01345795), (0.20381963, 0.030416), (0.38464387, 0.03351085),
        (0.54212002, 0.02482939), (0.72165694, 0.02741654), (0.88611635, 0.03719118), (1.0, 0.00339958)),
}
# msxFI FeFET: gamma (shape, loc, scale) per level, amperes, lowest first.
FEFET_MSXFI = {
    2: ((0.23561349688784233, 2.4454999999999995e-13, 2.1169150466830806e-07),
        (2.212466378535213, 3.975885307814147e-06, 1.441335439013419e-07)),
    4: ((1.273596301270418, 1.4989607330505366e-07, 1.5298418930548967e-07),
        (1.3206904391777872, 1.440137691539061e-06, 1.7501566349401908e-07),
        (1.3535503722473416, 2.7192534468261793e-06, 1.7737875476545208e-07),
        (2.028297515271994, 3.982059133364551e-06, 1.5660096428872716e-07)),
    8: ((0.23934594198369086, 2.4454999999999995e-13, 2.1457435169483346e-07),
        (1.3052793853280775, 1.4964348815637574e-07, 1.4980311940478568e-07),
        (1.36137001075976, 7.991625161922838e-07, 1.7617219105787133e-07),
        (1.3724682802336625, 1.4393150070067448e-06, 1.700883616073342e-07),
        (1.3007816431308712, 2.0798491605766206e-06, 1.5734580105177185e-07),
        (1.2870147526012115, 2.7207302342417214e-06, 1.937330139816702e-07),
        (1.434610638402805, 3.3590383705953837e-06, 1.5716953520934526e-07),
        (1.9428705913618374, 3.983588735245695e-06, 1.5585905645890258e-07)),
}
MSXFI_FILES = {"rram": "mlc_rram_args.p (sha256 6f734055c1e4036b...)", "fefet": "mlc_fefet_args.p (sha256 bbd93a7573a53660...)"}


@dataclass(frozen=True)
class LevelFaultModel:
    """Chance that a cell stored at each physical level is read one level down or up."""

    name: str
    p_down: tuple
    p_up: tuple
    parameters: dict = field(default_factory=dict)

    @property
    def num_levels(self):
        return len(self.p_down)

    @property
    def fault_probability(self):
        return tuple(d + u for d, u in zip(self.p_down, self.p_up))

    def validate(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Fault model needs a name")
        k = len(self.p_down)
        if k < 2 or len(self.p_up) != k:
            raise ValueError("Fault model needs matching p_down/p_up for at least two levels")
        for value in self.p_down + self.p_up:
            if isinstance(value, bool) or not isinstance(value, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Level fault probabilities must be finite floats in [0, 1]")
        if self.p_down[0] != 0 or self.p_up[-1] != 0:
            raise ValueError("Level 0 cannot move down and the last level cannot move up")
        if any(d + u > 1 for d, u in zip(self.p_down, self.p_up)):
            raise ValueError("A level's total fault probability cannot exceed 1")

    def to_dict(self):
        return {"name": self.name, "num_levels": self.num_levels, "p_down": list(self.p_down),
                "p_up": list(self.p_up), "parameters": dict(self.parameters)}

    @classmethod
    def from_dict(cls, data):
        model = cls(data["name"], tuple(float(v) for v in data["p_down"]), tuple(float(v) for v in data["p_up"]),
                    dict(data.get("parameters", {})))
        if data.get("num_levels") != model.num_levels:
            raise ValueError("Saved fault model level count differs from its probability tables")
        model.validate()
        return model


# ---------- distributions and thresholds ----------

def _gauss_below(x, mean, sigma):
    return 0.5 * math.erfc((mean - x) / (sigma * math.sqrt(2)))


def _gauss_above(x, mean, sigma):
    return 0.5 * math.erfc((x - mean) / (sigma * math.sqrt(2)))


def _gamma_below(x, shape, loc, scale):
    z = (x - loc) / scale
    return 0.0 if z <= 0 else float(torch.special.gammainc(torch.tensor(shape, dtype=torch.float64), torch.tensor(z, dtype=torch.float64)))


def _gamma_above(x, shape, loc, scale):
    z = (x - loc) / scale
    return 1.0 if z <= 0 else float(torch.special.gammaincc(torch.tensor(shape, dtype=torch.float64), torch.tensor(z, dtype=torch.float64)))


def _gamma_median(shape, loc, scale):
    lo, hi = loc, loc + scale * (shape + 40 * math.sqrt(shape) + 40)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if _gamma_below(mid, shape, loc, scale) < 0.5 else (lo, mid)
    return 0.5 * (lo + hi)


def _interpolate(x, xs, ys):
    """Piecewise-linear interpolation, clamped to the end values (xs increasing)."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for j in range(len(xs) - 1):
        if xs[j] <= x <= xs[j + 1]:
            return ys[j] + (ys[j + 1] - ys[j]) * (x - xs[j]) / (xs[j + 1] - xs[j])
    raise AssertionError("unreachable")


def equal_tail_threshold(a, b):
    """Point the same number of sigmas from both Gaussian means (the midpoint when sigmas match)."""
    (ma, sa), (mb, sb) = a, b
    return (ma * sb + mb * sa) / (sa + sb)


def intersection_threshold(a, b):
    """Where the two Gaussian densities cross between the means (msxFI's RRAM rule)."""
    (m1, s1), (m2, s2) = a, b
    qa = 1 / (2 * s1 ** 2) - 1 / (2 * s2 ** 2)
    qb = m2 / s2 ** 2 - m1 / s1 ** 2
    qc = m1 ** 2 / (2 * s1 ** 2) - m2 ** 2 / (2 * s2 ** 2) - math.log(s2 / s1)
    if abs(qa) < 1e-12 * max(abs(qb), 1.0):
        roots = [-qc / qb]
    else:
        disc = math.sqrt(max(qb * qb - 4 * qa * qc, 0.0))
        roots = [(-qb + disc) / (2 * qa), (-qb - disc) / (2 * qa)]
    lo, hi = sorted((m1, m2))
    inside = [r for r in roots if lo < r < hi]
    if not inside:
        raise ValueError("Gaussian densities do not cross between the level means")
    return inside[0]


def _model_from_levels(name, levels, below, above, threshold, parameters):
    """Build p_down/p_up for levels listed in physical order with strictly monotone centres."""
    centres = [lv[0] if len(lv) == 2 else _gamma_median(*lv) for lv in levels]
    increasing = all(b > a for a, b in zip(centres, centres[1:]))
    decreasing = all(b < a for a, b in zip(centres, centres[1:]))
    if not (increasing or decreasing):
        raise ValueError("Level centres must be strictly monotone in physical order")
    k = len(levels)
    p_down, p_up = [0.0] * k, [0.0] * k
    for i in range(k - 1):
        t = threshold(levels[i], levels[i + 1])
        if increasing:   # the next level has a larger signal
            p_up[i], p_down[i + 1] = above(t, *levels[i]), below(t, *levels[i + 1])
        else:            # the next level has a smaller signal (CTT)
            p_up[i], p_down[i + 1] = below(t, *levels[i]), above(t, *levels[i + 1])
    model = LevelFaultModel(name, tuple(float(v) for v in p_down), tuple(float(v) for v in p_up), parameters)
    model.validate()
    return model


# ---------- presets ----------

def ctt_levels(num_levels, *, chip="maxnvm2019", l0_gap_ua=None, sigma_prog_ua=None, sigma_mode="flat", read_noise_ua=0.0):
    """Per-level (mean, sigma) in uA for a CTT cell with ``num_levels`` levels.

    With the chip's own level count and no overrides, the measured levels are
    used as they are. Otherwise the levels are rebuilt in the chip's measured
    current window: L0 as measured, L1 ``l0_gap_ua`` below it (default: the
    measured L0-L1 gap), and the remaining programmed levels evenly spaced down
    to the chip's lowest measured level. Programmed levels get one sigma
    (``sigma_mode="flat"``; default the mean measured programmed sigma above
    10 uA) or a sigma interpolated by current from the measured levels
    (``"interpolated"``). ``read_noise_ua`` is added in quadrature to every level.
    """
    if chip not in CTT_CHIPS:
        raise ValueError(f"Unknown CTT chip {chip!r}; choose from {sorted(CTT_CHIPS)}")
    if isinstance(num_levels, bool) or not isinstance(num_levels, int) or not 2 <= num_levels <= 256:
        raise ValueError("num_levels must be an integer from 2 to 256")
    for label, value in (("l0_gap_ua", l0_gap_ua), ("sigma_prog_ua", sigma_prog_ua)):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{label} must be positive")
    if not math.isfinite(read_noise_ua) or read_noise_ua < 0:
        raise ValueError("read_noise_ua must be non-negative")
    if sigma_mode not in ("flat", "interpolated"):
        raise ValueError("sigma_mode must be 'flat' or 'interpolated'")
    measured = CTT_CHIPS[chip]
    if num_levels == len(measured) and l0_gap_ua is None and sigma_prog_ua is None:
        levels = list(measured)
    else:
        (m0, s0), programmed = measured[0], measured[1:]
        top = m0 - (m0 - programmed[0][0] if l0_gap_ua is None else l0_gap_ua)
        bottom = programmed[-1][0]
        if num_levels > 2 and not top > bottom:
            raise ValueError("l0_gap_ua leaves no room for the programmed levels")
        means = [top] if num_levels == 2 else [top + (bottom - top) * i / (num_levels - 2) for i in range(num_levels - 1)]
        if sigma_prog_ua is not None:
            sigmas = [sigma_prog_ua] * len(means)
        elif sigma_mode == "flat":
            above = [s for m, s in programmed if m >= NEAR_TURN_ON_UA]
            sigmas = [sum(above) / len(above)] * len(means)
        else:
            xs, ys = [m for m, _ in programmed][::-1], [s for _, s in programmed][::-1]   # increasing current
            sigmas = [_interpolate(m, xs, ys) for m in means]
        levels = [(m0, s0)] + list(zip(means, sigmas))
    return [(m, math.hypot(s, read_noise_ua)) for m, s in levels]


def ctt_model(num_levels, *, chip="maxnvm2019", l0_gap_ua=None, sigma_prog_ua=None, sigma_mode="flat", read_noise_ua=0.0):
    """CTT fault model from measured chip data; equal-tail thresholds between neighbours."""
    levels = ctt_levels(num_levels, chip=chip, l0_gap_ua=l0_gap_ua, sigma_prog_ua=sigma_prog_ua,
                        sigma_mode=sigma_mode, read_noise_ua=read_noise_ua)
    measured_as_is = num_levels == len(CTT_CHIPS[chip]) and l0_gap_ua is None and sigma_prog_ua is None
    parameters = {"technology": "CTT", "chip": chip, "signal": "read current at 0.8 V (uA)",
                  "levels_source": "measured" if measured_as_is else "rebuilt in the measured current window",
                  "l0_gap_ua": l0_gap_ua, "sigma_prog_ua": sigma_prog_ua, "sigma_mode": sigma_mode,
                  "read_noise_ua": read_noise_ua, "threshold_rule": "equal_tail",
                  "level_means_ua": [round(m, 4) for m, _ in levels], "level_sigmas_ua": [round(s, 4) for _, s in levels]}
    name = f"ctt_{chip}_{num_levels}lvl" + (f"_noise{read_noise_ua:g}ua" if read_noise_ua else "")
    return _model_from_levels(name, levels, _gauss_below, _gauss_above, equal_tail_threshold, parameters)


def msxfi_model(technology, num_levels):
    """RRAM or FeFET fault model from the msxFI distribution files (2, 4 or 8 levels)."""
    if technology == "rram":
        table, below, above, threshold, rule = RRAM_MSXFI, _gauss_below, _gauss_above, intersection_threshold, "gaussian_intersection"
    elif technology == "fefet":
        table, below, above, rule = FEFET_MSXFI, _gamma_below, _gamma_above, "median_midpoint"
        threshold = lambda a, b: 0.5 * (_gamma_median(*a) + _gamma_median(*b))
    else:
        raise ValueError("technology must be 'rram' or 'fefet'")
    if num_levels not in table:
        raise ValueError(f"msxFI {technology} data covers {sorted(table)} levels, not {num_levels}")
    parameters = {"technology": technology.upper() if technology == "rram" else "FeFET", "source": f"msxFI {MSXFI_FILES[technology]}",
                  "distribution": "gaussian" if technology == "rram" else "gamma", "threshold_rule": rule,
                  "read_circuit": "unknown whether included"}
    return _model_from_levels(f"{technology}_msxfi_{num_levels}lvl", list(table[num_levels]), below, above, threshold, parameters)


DEVICES = ("ctt_maxnvm2019", "ctt_dac2018", "rram_msxfi", "fefet_msxfi")


def device_model(device, num_levels, *, read_noise_ua=0.0, l0_gap_ua=None, sigma_prog_ua=None, sigma_mode="flat"):
    """Build a named device's fault model for ``num_levels`` levels."""
    if device.startswith("ctt_"):
        return ctt_model(num_levels, chip=device[4:], l0_gap_ua=l0_gap_ua, sigma_prog_ua=sigma_prog_ua,
                         sigma_mode=sigma_mode, read_noise_ua=read_noise_ua)
    if device in ("rram_msxfi", "fefet_msxfi"):
        if read_noise_ua or l0_gap_ua is not None or sigma_prog_ua is not None or sigma_mode != "flat":
            raise ValueError("Read noise, L0 gap and sigma overrides apply to CTT devices only")
        return msxfi_model(device.split("_")[0], num_levels)
    raise ValueError(f"Unknown device {device!r}; choose from {DEVICES}")
