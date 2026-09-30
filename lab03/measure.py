from __future__ import annotations

import statistics
from typing import Any

from bench import Bench, measured, read_first, read_text, unknown

import json

# A sample is still warm-up while it exceeds the settled rate by this fraction.
WARMUP_TOL = 0.5

# How many samples must sit strictly above a quantile before that quantile is an
# estimate rather than "the biggest number we saw, wearing a hat".
MIN_SAMPLES_ABOVE = 5

# Percentiles the record carries, in the order the schema lists them.
PERCENTILES = (50, 95, 99)

# The widest gap between neighbouring measurements, as a multiple of the typical
# gap, beyond which the sample is treated as coming from two populations.
MULTIMODAL_GAP_RATIO = 20.0

# Neither side of that gap is a mode unless it holds at least this fraction.
MIN_MODE_FRACTION = 0.10

# Below this many retained samples, modality is not a question worth answering.
MIN_SAMPLES_FOR_MODALITY = 20

# How far the last third of a run may drift from the first third, relative to
# the run's own median, before the run is not one population either.
STATIONARITY_TOL = 0.10
MIN_SAMPLES_FOR_STATIONARITY = 12

THERMAL_ZONES = "sys/devices/virtual/thermal"

POWER_RAIL_CANDIDATES = (
    "sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/in1_input",
    "sys/bus/i2c/drivers/ina3221/1-0040/iio:device0/in_power0_input",
    "sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
)

GPU_LOAD_CANDIDATES = (
    "sys/devices/platform/gpu.0/load",
    "sys/devices/gpu.0/load",
)

CPUFREQ_MIN = "sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq"
CPUFREQ_MAX = "sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"


# ===========================================================================
# 1. The loop
# ===========================================================================
def run_timed_iterations(bench: Bench, repeats: int = 100) -> list[float]:
    samples: list[float] = []
    for _ in range(repeats):
        bench.workload.synchronize()
        start = bench.clock()
        bench.workload.run()
        bench.workload.synchronize()
        end = bench.clock()
        samples.append((end - start) / 1_000_000)
    return samples


def _warmup_source() -> str:
    return (
        f"leading prefix above (1 + {WARMUP_TOL}) x median of the run's second half"
    )


def _is_drifting(samples: list[float]) -> bool:
    third = len(samples) // 3
    if third == 0:
        return False
    first = statistics.median(samples[:third])
    last = statistics.median(samples[-third:])
    med = statistics.median(samples)
    if med == 0:
        return first != last
    return abs(last - first) / med > STATIONARITY_TOL


def find_warmup_boundary(samples: list[float]) -> dict[str, Any]:
    source = _warmup_source()
    if not samples:
        return unknown(source, "no samples to compare against a settled tail")
    settled = statistics.median(samples[len(samples) // 2 :])
    threshold = (1 + WARMUP_TOL) * settled
    discarded = 0
    for sample in samples:
        if sample > threshold:
            discarded += 1
        else:
            break
    if len(samples) >= MIN_SAMPLES_FOR_STATIONARITY and _is_drifting(samples):
        discarded = 0
    return {
        "value": discarded,
        "source": source,
        "status": "ok",
        "settled_rate_ms": round(settled, 4),
        "threshold_ms": round(threshold, 4),
        "tolerance": WARMUP_TOL,
        "retained": len(samples) - discarded,
    }


def _percentile(ordered: list[float], percent: int) -> float:
    n = len(ordered)
    if n == 1:
        return ordered[0]
    pos = (n - 1) * (percent / 100.0)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def summarize(samples: list[float]) -> dict[str, Any]:
    n = len(samples)
    out: dict[str, Any] = {"n": n}
    if n == 0:
        out["mean"] = None
        out["std"] = None
        out["min"] = None
        out["max"] = None
        for percent in PERCENTILES:
            out[f"p{percent}"] = None
        return out
    ordered = sorted(samples)
    std = statistics.stdev(samples) if n >= 2 else 0.0
    out["mean"] = round(statistics.mean(samples), 4)
    out["std"] = round(std, 4)
    out["min"] = round(ordered[0], 4)
    out["max"] = round(ordered[-1], 4)
    for percent in PERCENTILES:
        out[f"p{percent}"] = round(_percentile(ordered, percent), 4)
    return out


def _modality_source() -> str:
    percent = int(round(MIN_MODE_FRACTION * 100))
    return (
        f"widest trimmed gap >= {MULTIMODAL_GAP_RATIO}x the median gap, "
        f"with >= {percent}% of samples on each side"
    )


def is_multimodal(samples: list[float]) -> dict[str, Any]:
    source = _modality_source()
    if len(samples) < MIN_SAMPLES_FOR_MODALITY:
        return unknown(
            source,
            f"need at least {MIN_SAMPLES_FOR_MODALITY} samples to judge modality",
        )
    ordered = sorted(samples)
    trim = MIN_SAMPLES_ABOVE
    body = ordered[trim:-trim] if len(ordered) > 2 * trim else ordered
    if len(body) < 2:
        return unknown(source, "not enough samples left after trimming the tails")
    gaps = [body[i + 1] - body[i] for i in range(len(body) - 1)]
    typical = statistics.median(gaps)
    widest_i = max(range(len(gaps)), key=lambda i: gaps[i])
    widest = gaps[widest_i]
    ratio = widest / typical if typical else (float("inf") if widest else 0.0)
    left_edge = body[widest_i]
    left = [sample for sample in ordered if sample <= left_edge]
    right = [sample for sample in ordered if sample > left_edge]
    total = len(samples)
    left_share = len(left) / total
    right_share = len(right) / total
    return {
        "value": ratio >= MULTIMODAL_GAP_RATIO
        and left_share >= MIN_MODE_FRACTION
        and right_share >= MIN_MODE_FRACTION,
        "source": source,
        "status": "ok",
        "gap_ratio": round(ratio, 2),
        "widest_gap_ms": round(widest, 3),
        "typical_gap_ms": round(typical, 5),
        "modes": [
            {
                "n": len(left),
                "share": left_share,
                "median_ms": round(statistics.median(left), 4) if left else None,
            },
            {
                "n": len(right),
                "share": right_share,
                "median_ms": round(statistics.median(right), 4) if right else None,
            },
        ],
    }


# ===========================================================================
# 7. The clock ceiling the run happened under
# ===========================================================================


def _clock_ceiling(bench: Bench) -> tuple[bool | None, dict[str, Any]]:
    source = f"{CPUFREQ_MIN} vs {CPUFREQ_MAX}"
    try:
        min_hz = read_text(bench.telemetry, CPUFREQ_MIN)
        max_hz = read_text(bench.telemetry, CPUFREQ_MAX)
    except TypeError:
        min_hz = None
        max_hz = None
    if min_hz is None or max_hz is None:
        return None, unknown(
            source, "scaling_min_freq or scaling_max_freq unreadable"
        )
    pinned = int(min_hz) == int(max_hz)
    return pinned, measured(
        f"scaling_min_freq={min_hz}, scaling_max_freq={max_hz}",
        source,
    )


def probe_power_state(bench: Bench) -> dict[str, Any]:
    src = "nvpmodel -q"
    result = bench.runner(["nvpmodel", "-q"])
    if not result.ok or result.returncode != 0 or not result.stdout.strip():
        why = result.error or "nvpmodel absent or returned nothing"
        if result.ok:
            why = "nvpmodel absent or returned nothing"
        return unknown(src, why)

    name = None
    mode_index = None
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("NV Power Mode:"):
            name = stripped.split(":", 1)[1].strip()
        elif stripped.lstrip("-").isdigit():
            mode_index = int(stripped)

    if not name:
        return unknown(src, "no 'NV Power Mode:' line in nvpmodel output")

    jetson_clocks, clocks = _clock_ceiling(bench)
    return {
        "value": name,
        "source": src,
        "status": "ok",
        "mode_index": mode_index,
        "jetson_clocks": jetson_clocks,
        "jetson_clocks_source": clocks,
    }


def _probe_temperature(bench: Bench) -> dict[str, Any]:
    src = "sys/devices/virtual/thermal/*/temp"
    base = bench.telemetry / THERMAL_ZONES
    try:
        found = list(base.glob("thermal_zone*"))
    except OSError:
        found = []

    numbered = []
    for zone in found:
        suffix = zone.name[len("thermal_zone") :]
        if suffix.isdigit():
            numbered.append((int(suffix), zone))
    numbered.sort()

    zones: list[tuple[float, str | None]] = []
    for _, zone in numbered:
        try:
            raw = read_text(bench.telemetry, f"{THERMAL_ZONES}/{zone.name}/temp")
        except TypeError:
            continue
        if not raw:
            continue
        try:
            temp_c = int(raw) / 1000.0
        except ValueError:
            continue
        try:
            zone_type = read_text(bench.telemetry, f"{THERMAL_ZONES}/{zone.name}/type")
        except TypeError:
            zone_type = None
        zones.append((temp_c, zone_type))

    if not zones:
        return unknown(src, "no thermal zone reported a readable temperature")

    temp_c, zone_type = max(zones, key=lambda item: item[0])
    return measured(temp_c, src, zone=zone_type, zones_read=len(zones))


def _probe_power(bench: Bench) -> dict[str, Any]:
    src = " | ".join(POWER_RAIL_CANDIDATES)
    try:
        found = read_first(bench.telemetry, POWER_RAIL_CANDIDATES)
    except TypeError:
        found = None
    if not found:
        return unknown(src, "none of the documented INA3221 rail paths could be read")
    path, text = found
    try:
        value = float(text)
    except ValueError:
        return unknown(path, f"power rail present but unparseable: {text}")
    return measured(value, path)


def _probe_gpu(bench: Bench) -> dict[str, Any]:
    src = " | ".join(GPU_LOAD_CANDIDATES)
    try:
        found = read_first(bench.telemetry, GPU_LOAD_CANDIDATES)
    except TypeError:
        found = None
    if not found:
        return unknown(src, "none of the documented GPU load paths could be read")
    path, text = found
    token = text.split()[0] if text.split() else ""
    try:
        value = float(token) / 10.0
    except ValueError:
        return unknown(path, f"GPU load present but unparseable: {text}")
    return measured(value, path, units="per-mille / 10")


def probe_telemetry(bench: Bench) -> dict[str, Any]:
    return {
        "temperature_c": _probe_temperature(bench),
        "power_mw": _probe_power(bench),
        "gpu_utilization_percent": _probe_gpu(bench),
    }

## for debugging - uncomment the following lines for debugging.
# if __name__ == "__main__":
    # env = Bench.real()
    # out = find_warmup_boundary(samples)
    # print(out)

# for generating system_report.json
if __name__ == "__main__":
    # calling base environment
    try:
        env = Bench.real()
    except Exception as exc:
        if type(exc).__name__ != "AcceleratorError":
            raise
        env = Bench.real(device="cpu")

    # get your samples
    samples = run_timed_iterations(env, repeats=100)

    # testing measurments and probes
    report = {
        "warmup_boundary": find_warmup_boundary(samples),
        "summarize_setup": summarize(samples),
        "is_multimodal": is_multimodal(samples),
        "probe_power_state": probe_power_state(env),
        "probe_telemetry": probe_telemetry(env),
    }

    # save samples
    path = "samples_analysis.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4)

    # save report
    path = "system_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)