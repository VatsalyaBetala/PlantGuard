"""Plant disease epidemiology models.

Pure functions. No I/O. Each takes hourly weather and returns a 0-100
pressure score. Sources cited inline. Use as documented estimates,
not authoritative regulatory advice.

Models implemented:
  - Beaumont period (late blight): Beaumont, 1947.
  - Wallin DSV (early blight / Alternaria): Madden, Pennypacker & MacNab, 1978.
  - Gubler-Thomas index (powdery mildew, grape): Gubler et al., 1999.
  - Fungal generic risk: logistic on RH-hours and temp optimum.
  - Reproductive number R_eff via Wallinga-Teunis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass
class HourlyWeather:
    """One hour of weather at one grid cell."""
    ts_iso: str          # ISO 8601 string, hourly resolution
    temp_c: float
    rh_pct: float        # 0-100
    rain_mm: float       # rain in this hour
    wind_kph: float = 0.0
    dewpoint_c: float | None = None

    @property
    def leaf_wet(self) -> bool:
        """Proxy: leaf wet if RH >= 90% or any rain."""
        return self.rh_pct >= 90.0 or self.rain_mm > 0.05


# ---------------------------------------------------------------------------
# Disease generation intervals (days). Used for R estimation.
# ---------------------------------------------------------------------------

GENERATION_INTERVAL_DAYS: dict[str, float] = {
    "late blight": 4.0,
    "early blight": 6.0,
    "powdery mildew": 7.0,
    "downy mildew": 5.0,
    "leaf mold": 8.0,
    "septoria leaf spot": 10.0,
    "spider mites": 7.0,
    "black rot": 12.0,
    "esca": 21.0,          # chronic, slow
    "mosaic virus": 14.0,
    "yellow leaf curl virus": 14.0,
    "bacterial spot": 7.0,
    "scab": 9.0,
    "default": 7.0,
}


def generation_interval(disease: str) -> float:
    key = disease.replace("_", " ").strip().lower()
    if key in GENERATION_INTERVAL_DAYS:
        return GENERATION_INTERVAL_DAYS[key]
    for known, days in GENERATION_INTERVAL_DAYS.items():
        if known in key or key in known:
            return days
    return GENERATION_INTERVAL_DAYS["default"]


# ---------------------------------------------------------------------------
# Beaumont period (late blight)
#
# A "Beaumont period" occurs when, for 2 consecutive days:
#   - min temperature >= 10 C
#   - relative humidity >= 75% for at least 11 hours per day
# Each Beaumont period adds ~50 to pressure (capped at 100).
# ---------------------------------------------------------------------------

def beaumont_score(hours: Sequence[HourlyWeather]) -> float:
    if len(hours) < 24:
        return 0.0
    days = _split_days(hours)
    if len(days) < 2:
        return 0.0

    qualifying = [_qualifies_beaumont(day) for day in days]
    pairs = sum(1 for i in range(len(qualifying) - 1) if qualifying[i] and qualifying[i + 1])

    score = min(100.0, pairs * 50.0)
    # Boost slightly if any day saw heavy rain — wash spore release.
    rain_today = sum(h.rain_mm for h in days[-1])
    if rain_today >= 5.0:
        score = min(100.0, score + 10.0)
    return score


def _qualifies_beaumont(day: Sequence[HourlyWeather]) -> bool:
    if not day:
        return False
    temps = [h.temp_c for h in day]
    if min(temps) < 10.0:
        return False
    rh_hours = sum(1 for h in day if h.rh_pct >= 75.0)
    return rh_hours >= 11


# ---------------------------------------------------------------------------
# Wallin DSV (Disease Severity Value, early blight)
#
# Lookup of (avg temp during leaf wetness, hours of leaf wetness) -> 0..4.
# Sum DSVs across recent days; >= 7 cumulative is treatment threshold.
# ---------------------------------------------------------------------------

# rows: hours of leaf wetness band -> dsv per temp band
# columns are temp bins (C): <13, 13-17, 17-21, 21-26, >26
_WALLIN_TABLE: dict[tuple[int, int], list[int]] = {
    (0, 6):   [0, 0, 0, 0, 0],
    (7, 12):  [0, 1, 1, 2, 1],
    (13, 18): [0, 2, 2, 3, 2],
    (19, 24): [0, 3, 3, 4, 3],
}


def _wallin_temp_bin(t: float) -> int:
    if t < 13:  return 0
    if t < 17:  return 1
    if t < 21:  return 2
    if t < 26:  return 3
    return 4


def _wallin_dsv_for_day(day: Sequence[HourlyWeather]) -> int:
    wet_hours = [h for h in day if h.leaf_wet]
    n_wet = len(wet_hours)
    if n_wet == 0:
        return 0
    avg_t = sum(h.temp_c for h in wet_hours) / n_wet
    bin_t = _wallin_temp_bin(avg_t)
    for (lo, hi), row in _WALLIN_TABLE.items():
        if lo <= n_wet <= hi:
            return row[bin_t]
    if n_wet > 24:
        return _WALLIN_TABLE[(19, 24)][bin_t]
    return 0


def wallin_score(hours: Sequence[HourlyWeather]) -> float:
    days = _split_days(hours)
    if not days:
        return 0.0
    cumulative = sum(_wallin_dsv_for_day(d) for d in days[-7:])  # rolling 7-day sum
    # Threshold for treatment is ~7. Map 0..14 -> 0..100.
    return float(min(100.0, (cumulative / 14.0) * 100.0))


# ---------------------------------------------------------------------------
# Gubler-Thomas powdery mildew index (grape).
#
# +20 per consecutive day with 6+ hours in 21-30 C.
# Reset to zero on any day with <6 such hours.
# ---------------------------------------------------------------------------

def gubler_score(hours: Sequence[HourlyWeather]) -> float:
    days = _split_days(hours)
    score = 0.0
    streak = 0
    for day in days:
        opt = sum(1 for h in day if 21.0 <= h.temp_c <= 30.0)
        if opt >= 6:
            streak += 1
        else:
            streak = max(0, streak - 1)
        score = min(100.0, streak * 20.0)
    return score


# ---------------------------------------------------------------------------
# Generic fungal logistic model. Fallback for diseases without a named index.
#
# Inputs combined: rolling 5-day fraction of hours with RH>=85% AND
# temperature in [optimum-5, optimum+5]. Squashed via logistic.
# ---------------------------------------------------------------------------

DISEASE_TEMP_OPTIMUM_C: dict[str, float] = {
    "late blight": 18.0,
    "early blight": 24.0,
    "septoria leaf spot": 22.0,
    "leaf mold": 22.0,
    "powdery mildew": 25.0,
    "black rot": 26.0,
    "esca": 25.0,
    "scab": 17.0,
    "bacterial spot": 28.0,
    "default": 22.0,
}


def _disease_temp_optimum(disease: str) -> float:
    key = disease.replace("_", " ").strip().lower()
    if key in DISEASE_TEMP_OPTIMUM_C:
        return DISEASE_TEMP_OPTIMUM_C[key]
    for known, t in DISEASE_TEMP_OPTIMUM_C.items():
        if known in key or key in known:
            return t
    return DISEASE_TEMP_OPTIMUM_C["default"]


def generic_fungal_score(hours: Sequence[HourlyWeather], disease: str) -> float:
    if not hours:
        return 0.0
    recent = hours[-24 * 5 :]  # last 5 days
    if not recent:
        return 0.0
    optimum = _disease_temp_optimum(disease)
    favorable = sum(
        1 for h in recent
        if h.rh_pct >= 85.0 and abs(h.temp_c - optimum) <= 5.0
    )
    fraction = favorable / len(recent)
    # logistic centered at fraction=0.25 (a quarter of recent hours favorable -> 50%).
    score = 1.0 / (1.0 + math.exp(-12.0 * (fraction - 0.25)))
    return float(min(100.0, score * 100.0))


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def pressure_score(hours: Sequence[HourlyWeather], disease: str) -> dict[str, float | str]:
    """Return the best-fit pressure score for ``disease`` plus the model used."""
    key = disease.replace("_", " ").strip().lower()
    if "late blight" in key:
        return {"score": beaumont_score(hours), "model": "beaumont_period"}
    if "early blight" in key:
        return {"score": wallin_score(hours), "model": "wallin_dsv"}
    if "powdery mildew" in key:
        return {"score": gubler_score(hours), "model": "gubler_thomas"}
    return {"score": generic_fungal_score(hours, disease), "model": "generic_fungal"}


# ---------------------------------------------------------------------------
# Reproductive number — Wallinga-Teunis on case timestamps.
#
# Estimate R as ratio of secondary cases to expected primary cases assuming
# generation interval gamma-distributed around `mean_g` days.
# Light, deliberately approximate.
# ---------------------------------------------------------------------------

def estimate_r_effective(
    case_timestamps_days: Sequence[float],
    mean_g_days: float,
) -> float | None:
    """Estimate R_eff from a list of case ordinal-day timestamps.

    Returns None if too few cases. Uses simple ratio of cases in the most
    recent generation window vs. the prior generation window.
    """
    if len(case_timestamps_days) < 4:
        return None
    times = sorted(case_timestamps_days)
    t_max = times[-1]
    g = max(1.0, mean_g_days)
    recent = sum(1 for t in times if t > t_max - g)
    prior = sum(1 for t in times if t_max - 2 * g < t <= t_max - g)
    if prior == 0:
        return None
    r = recent / prior
    # Clamp to plausible band.
    return float(max(0.0, min(r, 10.0)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_days(hours: Sequence[HourlyWeather]) -> list[list[HourlyWeather]]:
    """Split hourly samples into per-day buckets by date prefix of ts_iso."""
    by_day: dict[str, list[HourlyWeather]] = {}
    order: list[str] = []
    for h in hours:
        day_key = h.ts_iso[:10]
        if day_key not in by_day:
            by_day[day_key] = []
            order.append(day_key)
        by_day[day_key].append(h)
    return [by_day[k] for k in order]


def aggregate_hours(rows: Iterable[dict]) -> list[HourlyWeather]:
    """Convert Open-Meteo hourly response rows to HourlyWeather list."""
    out: list[HourlyWeather] = []
    for row in rows:
        out.append(
            HourlyWeather(
                ts_iso=row["time"],
                temp_c=float(row.get("temperature_2m") or 0.0),
                rh_pct=float(row.get("relative_humidity_2m") or 0.0),
                rain_mm=float(row.get("precipitation") or 0.0),
                wind_kph=float(row.get("wind_speed_10m") or 0.0),
                dewpoint_c=row.get("dew_point_2m"),
            )
        )
    return out
