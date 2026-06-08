"""Thin repository for predictions.

main.py talks only to these functions — never to SQLAlchemy directly.
That keeps the swap to Postgres later mechanical.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import and_, delete, select

from plant_disease.db import session_scope
from plant_disease.db.models import Outbreak, PressureScore, Prediction
from plant_disease.geo import grid_bucket, haversine_km, jitter_coords


def record_prediction(
    filename: str,
    *,
    original_name: str = "",
    plant: str = "",
    disease: str = "",
    heatmap: str | None = None,
    plant_conf: float | None = None,
    disease_conf: float | None = None,
    top3_json: str | None = None,
    backend: str = "cnn_resnet50",
    source: str = "upload",
    lat: float | None = None,
    lon: float | None = None,
    geo_accuracy_m: float | None = None,
    geo_source: str | None = None,
    capture_ts: datetime | None = None,
) -> dict:
    """Insert (or upsert by filename) a prediction row. Returns the canonical dict."""
    with session_scope() as session:
        existing = session.execute(
            select(Prediction).where(Prediction.filename == filename)
        ).scalar_one_or_none()

        if existing is None:
            row = Prediction(
                filename=filename,
                original_name=original_name,
                plant=plant,
                disease=disease,
                heatmap=heatmap,
                plant_conf=plant_conf,
                disease_conf=disease_conf,
                top3_json=top3_json,
                backend=backend,
                source=source,
                lat=lat,
                lon=lon,
                geo_accuracy_m=geo_accuracy_m,
                geo_source=geo_source,
                capture_ts=capture_ts,
            )
            session.add(row)
            session.flush()
        else:
            existing.original_name = original_name or existing.original_name
            existing.plant = plant
            existing.disease = disease
            existing.heatmap = heatmap
            existing.plant_conf = plant_conf
            existing.disease_conf = disease_conf
            existing.top3_json = top3_json
            existing.backend = backend
            existing.source = source
            if lat is not None:
                existing.lat = lat
            if lon is not None:
                existing.lon = lon
            if geo_accuracy_m is not None:
                existing.geo_accuracy_m = geo_accuracy_m
            if geo_source is not None:
                existing.geo_source = geo_source
            if capture_ts is not None:
                existing.capture_ts = capture_ts
            row = existing

        return row.to_dict()


def list_predictions() -> list[dict]:
    """Return every prediction, newest first."""
    with session_scope() as session:
        rows = session.execute(
            select(Prediction).order_by(Prediction.created_at.desc(), Prediction.id.desc())
        ).scalars().all()
        return [row.to_dict() for row in rows]


def get_prediction(filename: str) -> dict | None:
    with session_scope() as session:
        row = session.execute(
            select(Prediction).where(Prediction.filename == filename)
        ).scalar_one_or_none()
        return row.to_dict() if row else None


def delete_prediction(filename: str) -> bool:
    """Returns True if a row was deleted."""
    with session_scope() as session:
        result = session.execute(
            delete(Prediction).where(Prediction.filename == filename)
        )
        return bool(result.rowcount)


def delete_all_predictions() -> int:
    """Returns number of rows deleted."""
    with session_scope() as session:
        result = session.execute(delete(Prediction))
        return int(result.rowcount or 0)


# ---------------------------------------------------------------------------
# Geo / outbreak / pressure queries
# ---------------------------------------------------------------------------

def list_geo_predictions(
    *,
    since_days: int | None = 90,
    plant: str | None = None,
    disease: str | None = None,
    jitter: bool = True,
) -> list[dict]:
    """Return geo-tagged predictions for map rendering, optionally jittered."""
    with session_scope() as session:
        stmt = (
            select(Prediction)
            .where(Prediction.lat.isnot(None))
            .where(Prediction.lon.isnot(None))
        )
        if since_days is not None:
            cutoff = datetime.utcnow() - timedelta(days=since_days)
            stmt = stmt.where(Prediction.created_at >= cutoff)
        if plant:
            stmt = stmt.where(Prediction.plant == plant)
        if disease:
            stmt = stmt.where(Prediction.disease == disease)
        rows = session.execute(stmt).scalars().all()

        out: list[dict] = []
        for r in rows:
            lat, lon = r.lat, r.lon
            if jitter and lat is not None and lon is not None:
                lat, lon = jitter_coords(lat, lon, max_m=200.0, seed=hash(r.filename) & 0xFFFFFFFF)
            out.append(
                {
                    "filename": r.filename,
                    "plant": r.plant,
                    "disease": r.disease,
                    "lat": lat,
                    "lon": lon,
                    "geo_source": r.geo_source,
                    "capture_ts": r.capture_ts.isoformat() if r.capture_ts else None,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "outbreak_id": r.outbreak_id,
                    "disease_conf": r.disease_conf,
                }
            )
        return out


def list_outbreaks(*, active_only: bool = False) -> list[dict]:
    with session_scope() as session:
        stmt = select(Outbreak).order_by(Outbreak.member_count.desc())
        if active_only:
            cutoff = datetime.utcnow() - timedelta(days=14)
            stmt = stmt.where(Outbreak.ended_at >= cutoff)
        rows = session.execute(stmt).scalars().all()
        return [r.to_dict() for r in rows]


def get_outbreak(outbreak_id: int) -> dict | None:
    with session_scope() as session:
        ob = session.get(Outbreak, outbreak_id)
        return ob.to_dict() if ob else None


def list_outbreaks_near(lat: float, lon: float, radius_km: float = 10.0, days: int = 14) -> list[dict]:
    """Outbreaks whose centroid lies within `radius_km` and ended within `days`."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    with session_scope() as session:
        rows = session.execute(
            select(Outbreak).where(Outbreak.ended_at >= cutoff)
        ).scalars().all()
        out: list[dict] = []
        for ob in rows:
            d = haversine_km(lat, lon, ob.centroid_lat, ob.centroid_lon)
            if d <= radius_km:
                payload = ob.to_dict()
                payload["distance_km"] = round(d, 2)
                out.append(payload)
        out.sort(key=lambda x: x["distance_km"])
        return out


def upsert_pressure_score(
    *,
    lat: float,
    lon: float,
    plant: str,
    disease: str,
    day: str,
    score: float,
    model: str,
) -> None:
    lat_g, lon_g = grid_bucket(lat, lon)
    with session_scope() as session:
        existing = session.execute(
            select(PressureScore)
            .where(PressureScore.lat_grid == lat_g)
            .where(PressureScore.lon_grid == lon_g)
            .where(PressureScore.plant == plant)
            .where(PressureScore.disease == disease)
            .where(PressureScore.day == day)
        ).scalar_one_or_none()
        if existing is None:
            session.add(
                PressureScore(
                    lat_grid=lat_g,
                    lon_grid=lon_g,
                    plant=plant,
                    disease=disease,
                    day=day,
                    score=score,
                    model=model,
                )
            )
        else:
            existing.score = score
            existing.model = model
            existing.computed_at = datetime.utcnow()


def list_pressure_for_grid(lat: float, lon: float, *, disease: str | None = None) -> list[dict]:
    lat_g, lon_g = grid_bucket(lat, lon)
    with session_scope() as session:
        stmt = (
            select(PressureScore)
            .where(PressureScore.lat_grid == lat_g)
            .where(PressureScore.lon_grid == lon_g)
            .order_by(PressureScore.day.asc())
        )
        if disease:
            stmt = stmt.where(PressureScore.disease == disease)
        rows = session.execute(stmt).scalars().all()
        return [r.to_dict() for r in rows]


def list_all_pressure(*, day: str | None = None, disease: str | None = None) -> list[dict]:
    with session_scope() as session:
        stmt = select(PressureScore)
        if day:
            stmt = stmt.where(PressureScore.day == day)
        if disease:
            stmt = stmt.where(PressureScore.disease == disease)
        rows = session.execute(stmt).scalars().all()
        return [r.to_dict() for r in rows]


def attach_outbreak_to_prediction(filename: str, outbreak_id: int | None) -> None:
    with session_scope() as session:
        row = session.execute(
            select(Prediction).where(Prediction.filename == filename)
        ).scalar_one_or_none()
        if row:
            row.outbreak_id = outbreak_id


def timeline_buckets(
    *,
    since_days: int = 90,
    bucket: str = "day",
) -> list[dict]:
    """Per-day prediction counts split by healthy / diseased."""
    cutoff = datetime.utcnow() - timedelta(days=since_days)
    with session_scope() as session:
        rows = session.execute(
            select(Prediction).where(Prediction.created_at >= cutoff)
        ).scalars().all()
        by_day: dict[str, dict[str, int]] = {}
        for r in rows:
            t = r.capture_ts or r.created_at
            if t is None:
                continue
            key = t.date().isoformat()
            slot = by_day.setdefault(key, {"day": key, "healthy": 0, "diseased": 0, "total": 0})
            is_healthy = "healthy" in (r.disease or "").lower()
            slot["total"] += 1
            if is_healthy:
                slot["healthy"] += 1
            else:
                slot["diseased"] += 1
        return sorted(by_day.values(), key=lambda x: x["day"])
