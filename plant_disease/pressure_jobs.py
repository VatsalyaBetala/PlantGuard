"""Background job: compute disease-pressure scores for active outbreak grids.

Pulls weather for each grid cell that has had a recent prediction, runs
each named epidemiology model against the disease(s) seen there, and writes
day-bucketed scores to the `pressure_scores` table.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Iterable

from plant_disease.db.repo import (
    list_geo_predictions,
    list_outbreaks,
    upsert_pressure_score,
)
from plant_disease.epidemiology import aggregate_hours, pressure_score
from plant_disease.geo import grid_bucket
from plant_disease.weather import fetch_combined

logger = logging.getLogger("plant_disease.pressure_jobs")

# Diseases worth scoring on the public map. Healthy / Unknown excluded.
_SCORABLE_HINTS = (
    "blight", "spot", "rot", "mildew", "mold", "rust", "scab", "esca", "measles",
    "virus", "curl", "mosaic",
)


def _is_scorable(disease: str | None) -> bool:
    if not disease:
        return False
    d = disease.lower()
    if "healthy" in d or "unknown" in d:
        return False
    return any(hint in d for hint in _SCORABLE_HINTS)


def _active_grids() -> dict[tuple[float, float], dict[str, set[str]]]:
    """Map grid -> {plant: {disease, ...}} from recent geo-tagged predictions
    plus any outbreak centroid within window.
    """
    grids: dict[tuple[float, float], dict[str, set[str]]] = {}
    for row in list_geo_predictions(since_days=30, jitter=False):
        if not _is_scorable(row.get("disease")):
            continue
        g = grid_bucket(row["lat"], row["lon"])
        plant_map = grids.setdefault(g, {})
        plant_map.setdefault(row.get("plant") or "", set()).add(row["disease"])
    for ob in list_outbreaks(active_only=True):
        if not _is_scorable(ob.get("disease")):
            continue
        c = ob.get("centroid") or {}
        if c.get("lat") is None or c.get("lon") is None:
            continue
        g = grid_bucket(c["lat"], c["lon"])
        plant_map = grids.setdefault(g, {})
        plant_map.setdefault(ob.get("plant") or "", set()).add(ob["disease"])
    return grids


def _score_one_cell(
    lat: float,
    lon: float,
    plant: str,
    diseases: Iterable[str],
) -> int:
    """Fetch weather for the cell, score each disease day-by-day, persist."""
    rows = fetch_combined(lat, lon, past_days=14, forecast_days=7)
    if not rows:
        return 0
    hours = aggregate_hours(rows)
    if not hours:
        return 0

    # Score per (disease, day): roll up to day-end and feed all hours <= eod.
    by_day: dict[str, list] = {}
    for h in hours:
        by_day.setdefault(h.ts_iso[:10], []).append(h)
    days_sorted = sorted(by_day.keys())

    written = 0
    for disease in diseases:
        for i, day in enumerate(days_sorted):
            cumulative = []
            for d in days_sorted[: i + 1]:
                cumulative.extend(by_day[d])
            res = pressure_score(cumulative, disease)
            score = float(res.get("score", 0.0))
            model = str(res.get("model", "generic_fungal"))
            upsert_pressure_score(
                lat=lat,
                lon=lon,
                plant=plant,
                disease=disease,
                day=day,
                score=score,
                model=model,
            )
            written += 1
    return written


def compute_all_pressure() -> dict[str, int]:
    """Refresh pressure scores for every active grid cell. Returns counts."""
    cells = _active_grids()
    cell_count = 0
    score_count = 0
    for (lat, lon), plant_map in cells.items():
        for plant, diseases in plant_map.items():
            score_count += _score_one_cell(lat, lon, plant, diseases)
        cell_count += 1
    logger.info("Pressure refresh: %d cells, %d scores written", cell_count, score_count)
    return {"cells": cell_count, "scores_written": score_count}
