"""Open-Meteo client with grid-cell caching.

Free API, no key. Endpoints:
  - Forecast:    https://api.open-meteo.com/v1/forecast
  - Historical:  https://archive-api.open-meteo.com/v1/archive

Hourly variables fetched: temperature_2m, relative_humidity_2m,
precipitation, wind_speed_10m, dew_point_2m.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from sqlalchemy import select

from plant_disease.db import session_scope
from plant_disease.db.models import WeatherSnap
from plant_disease.geo import grid_bucket

logger = logging.getLogger("plant_disease.weather")

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARS = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "dew_point_2m",
)

CACHE_TTL_HOURS = 6


def _is_fresh(snap: WeatherSnap) -> bool:
    if snap.fetched_at is None:
        return False
    age = datetime.utcnow() - snap.fetched_at
    return age < timedelta(hours=CACHE_TTL_HOURS)


def _cache_get(lat_grid: float, lon_grid: float, kind: str) -> list[dict] | None:
    with session_scope() as session:
        snap = session.execute(
            select(WeatherSnap)
            .where(WeatherSnap.lat_grid == lat_grid)
            .where(WeatherSnap.lon_grid == lon_grid)
            .where(WeatherSnap.window_kind == kind)
            .order_by(WeatherSnap.fetched_at.desc())
        ).scalars().first()
        if snap is None or not _is_fresh(snap):
            return None
        try:
            return json.loads(snap.payload_json)
        except (TypeError, json.JSONDecodeError):
            return None


def _cache_put(lat_grid: float, lon_grid: float, kind: str, rows: list[dict]) -> None:
    with session_scope() as session:
        snap = WeatherSnap(
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            window_kind=kind,
            payload_json=json.dumps(rows),
        )
        session.add(snap)


def _rows_from_response(payload: dict[str, Any]) -> list[dict]:
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return []
    out: list[dict] = []
    for i, ts in enumerate(times):
        row = {"time": ts}
        for var in HOURLY_VARS:
            series = hourly.get(var) or []
            row[var] = series[i] if i < len(series) else None
        out.append(row)
    return out


def fetch_past_hours(lat: float, lon: float, days: int = 14) -> list[dict]:
    """Return hourly weather rows for the last `days`. Cached per grid cell."""
    lat_g, lon_g = grid_bucket(lat, lon)
    cached = _cache_get(lat_g, lon_g, "past")
    if cached is not None:
        return cached

    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    params = {
        "latitude": lat_g,
        "longitude": lon_g,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
    }
    try:
        resp = httpx.get(ARCHIVE_URL, params=params, timeout=20.0)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Open-Meteo archive fetch failed: %s", exc)
        return []
    rows = _rows_from_response(resp.json())
    _cache_put(lat_g, lon_g, "past", rows)
    return rows


def fetch_forecast_hours(lat: float, lon: float, days: int = 7) -> list[dict]:
    """Return hourly forecast rows for the next `days`. Cached per grid cell."""
    lat_g, lon_g = grid_bucket(lat, lon)
    cached = _cache_get(lat_g, lon_g, "forecast")
    if cached is not None:
        return cached

    params = {
        "latitude": lat_g,
        "longitude": lon_g,
        "forecast_days": days,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
    }
    try:
        resp = httpx.get(FORECAST_URL, params=params, timeout=20.0)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Open-Meteo forecast fetch failed: %s", exc)
        return []
    rows = _rows_from_response(resp.json())
    _cache_put(lat_g, lon_g, "forecast", rows)
    return rows


def fetch_combined(lat: float, lon: float, past_days: int = 14, forecast_days: int = 7) -> list[dict]:
    """Past hours followed by forecast hours, deduped by timestamp."""
    past = fetch_past_hours(lat, lon, days=past_days)
    fwd = fetch_forecast_hours(lat, lon, days=forecast_days)
    seen: set[str] = set()
    merged: list[dict] = []
    for row in past + fwd:
        ts = row.get("time")
        if not ts or ts in seen:
            continue
        seen.add(ts)
        merged.append(row)
    return merged
