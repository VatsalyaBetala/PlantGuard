"""Geo helpers: EXIF GPS extraction, grid bucketing, distance, jitter."""
from __future__ import annotations

import math
import random
from datetime import datetime
from typing import Tuple

try:
    import piexif
except Exception:  # pragma: no cover
    piexif = None

EARTH_RADIUS_KM = 6371.0088
GRID_STEP_DEG = 0.05  # ~5km. Used for weather + pressure caching.


# ---------------------------------------------------------------------------
# EXIF
# ---------------------------------------------------------------------------

def _ratio_to_float(value) -> float:
    """piexif returns rationals as (num, den) tuples."""
    if isinstance(value, tuple) and len(value) == 2:
        num, den = value
        if den == 0:
            return 0.0
        return num / den
    return float(value)


def _dms_to_decimal(dms, ref: bytes) -> float:
    if not dms or len(dms) != 3:
        return 0.0
    d = _ratio_to_float(dms[0])
    m = _ratio_to_float(dms[1])
    s = _ratio_to_float(dms[2])
    decimal = d + (m / 60.0) + (s / 3600.0)
    if ref in (b"S", b"W", "S", "W"):
        decimal = -decimal
    return decimal


def parse_exif_gps(image_path: str) -> dict | None:
    """Extract GPS + capture timestamp from EXIF. Returns None if absent."""
    if piexif is None:
        return None
    try:
        exif = piexif.load(image_path)
    except Exception:
        return None

    gps = exif.get("GPS") or {}
    if not gps:
        return None

    lat_dms = gps.get(piexif.GPSIFD.GPSLatitude)
    lat_ref = gps.get(piexif.GPSIFD.GPSLatitudeRef, b"N")
    lon_dms = gps.get(piexif.GPSIFD.GPSLongitude)
    lon_ref = gps.get(piexif.GPSIFD.GPSLongitudeRef, b"E")

    if not lat_dms or not lon_dms:
        return None

    lat = _dms_to_decimal(lat_dms, lat_ref)
    lon = _dms_to_decimal(lon_dms, lon_ref)
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None

    capture_ts: datetime | None = None
    exif_dt = (exif.get("Exif") or {}).get(piexif.ExifIFD.DateTimeOriginal)
    if exif_dt:
        try:
            ts_str = exif_dt.decode("utf-8") if isinstance(exif_dt, bytes) else exif_dt
            capture_ts = datetime.strptime(ts_str, "%Y:%m:%d %H:%M:%S")
        except (ValueError, AttributeError):
            capture_ts = None

    return {
        "lat": lat,
        "lon": lon,
        "capture_ts": capture_ts,
        "source": "exif",
    }


# ---------------------------------------------------------------------------
# Grid bucketing for caching weather/pressure.
# ---------------------------------------------------------------------------

def grid_bucket(lat: float, lon: float, step: float = GRID_STEP_DEG) -> Tuple[float, float]:
    return (
        round(round(lat / step) * step, 4),
        round(round(lon / step) * step, 4),
    )


# ---------------------------------------------------------------------------
# Privacy: jitter for public map display.
# ---------------------------------------------------------------------------

def jitter_coords(lat: float, lon: float, max_m: float = 200.0, seed: int | None = None) -> Tuple[float, float]:
    """Add deterministic jitter (with seed) up to max_m meters."""
    rng = random.Random(seed)
    deg_per_m_lat = 1.0 / 111_000.0
    deg_per_m_lon = 1.0 / (111_000.0 * max(0.1, math.cos(math.radians(lat))))
    dx = rng.uniform(-max_m, max_m)
    dy = rng.uniform(-max_m, max_m)
    return lat + dy * deg_per_m_lat, lon + dx * deg_per_m_lon


# ---------------------------------------------------------------------------
# Haversine.
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c
