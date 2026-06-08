"""Spatiotemporal clustering of diseased predictions into outbreaks.

ST-DBSCAN: a point belongs to a cluster if there exist >= minPts other points
within ε_space km AND within ε_time days. Sklearn lacks a direct ST-DBSCAN,
so we implement the canonical neighbor-graph variant on top of haversine
distance + day-delta.

Outputs are persisted to the `outbreaks` table; member predictions get
`outbreak_id` set.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select, update

from plant_disease.db import session_scope
from plant_disease.db.models import Outbreak, Prediction
from plant_disease.epidemiology import estimate_r_effective, generation_interval
from plant_disease.geo import haversine_km

logger = logging.getLogger("plant_disease.clustering")

EPS_SPACE_KM_DEFAULT = 2.0
EPS_TIME_DAYS_DEFAULT = 7.0
MIN_PTS_DEFAULT = 3


def _is_diseased(disease: str | None) -> bool:
    if not disease:
        return False
    d = disease.lower()
    return d not in {"unknown", "healthy"} and "healthy" not in d


def _days_between(a: datetime, b: datetime) -> float:
    return abs((a - b).total_seconds()) / 86400.0


def _convex_hull_geojson(points: list[tuple[float, float]]) -> str | None:
    """Compute convex hull of (lat, lon) points and return as GeoJSON Polygon."""
    if len(points) < 3:
        return None
    try:
        from shapely.geometry import MultiPoint, mapping
    except ImportError:
        return None
    # Shapely uses (lon, lat) ordering for GeoJSON.
    multi = MultiPoint([(lon, lat) for lat, lon in points])
    hull = multi.convex_hull
    if hull.is_empty:
        return None
    return json.dumps(mapping(hull))


def _cluster_one_disease(
    rows: list[Prediction],
    eps_km: float,
    eps_days: float,
    min_pts: int,
) -> list[list[Prediction]]:
    """Run ST-DBSCAN on a homogeneous-disease list. Returns clusters of >= min_pts."""
    n = len(rows)
    if n < min_pts:
        return []

    visited = [False] * n
    cluster_id = [-1] * n  # -1 = noise
    next_cid = 0

    def neighbors(i: int) -> list[int]:
        out: list[int] = []
        ri = rows[i]
        ti = ri.capture_ts or ri.created_at
        for j in range(n):
            if j == i:
                continue
            rj = rows[j]
            tj = rj.capture_ts or rj.created_at
            if _days_between(ti, tj) > eps_days:
                continue
            if haversine_km(ri.lat, ri.lon, rj.lat, rj.lon) <= eps_km:
                out.append(j)
        return out

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        nbrs = neighbors(i)
        if len(nbrs) + 1 < min_pts:
            continue
        cid = next_cid
        next_cid += 1
        cluster_id[i] = cid
        seeds = list(nbrs)
        while seeds:
            j = seeds.pop()
            if not visited[j]:
                visited[j] = True
                nbrs_j = neighbors(j)
                if len(nbrs_j) + 1 >= min_pts:
                    for k in nbrs_j:
                        if k not in seeds:
                            seeds.append(k)
            if cluster_id[j] == -1:
                cluster_id[j] = cid

    clusters: dict[int, list[Prediction]] = {}
    for i, cid in enumerate(cluster_id):
        if cid == -1:
            continue
        clusters.setdefault(cid, []).append(rows[i])
    return [members for members in clusters.values() if len(members) >= min_pts]


def recompute_outbreaks(
    *,
    eps_km: float = EPS_SPACE_KM_DEFAULT,
    eps_days: float = EPS_TIME_DAYS_DEFAULT,
    min_pts: int = MIN_PTS_DEFAULT,
    horizon_days: int = 90,
) -> dict[str, Any]:
    """Re-cluster all geo-tagged diseased predictions in the recent window.

    Replaces existing outbreaks. Idempotent — call from scheduler or admin route.
    Returns summary dict with counts and the list of new outbreak ids.
    """
    cutoff = datetime.utcnow() - timedelta(days=horizon_days)
    with session_scope() as session:
        # Wipe existing memberships + outbreak rows for clean re-cluster.
        session.execute(update(Prediction).values(outbreak_id=None))
        session.query(Outbreak).delete()
        session.flush()

        rows = session.execute(
            select(Prediction)
            .where(Prediction.lat.isnot(None))
            .where(Prediction.lon.isnot(None))
            .where(Prediction.created_at >= cutoff)
        ).scalars().all()

        # Filter for diseased only; group by disease.
        by_disease: dict[str, list[Prediction]] = {}
        for r in rows:
            if not _is_diseased(r.disease):
                continue
            by_disease.setdefault(r.disease.strip(), []).append(r)

        new_outbreak_ids: list[int] = []
        for disease, members in by_disease.items():
            clusters = _cluster_one_disease(members, eps_km, eps_days, min_pts)
            for cluster in clusters:
                ob = _persist_outbreak(session, disease, cluster)
                new_outbreak_ids.append(ob.id)

        return {
            "ok": True,
            "outbreak_count": len(new_outbreak_ids),
            "outbreak_ids": new_outbreak_ids,
            "considered_predictions": len(rows),
            "params": {"eps_km": eps_km, "eps_days": eps_days, "min_pts": min_pts},
        }


def _persist_outbreak(session, disease: str, cluster: list[Prediction]) -> Outbreak:
    lats = [p.lat for p in cluster]
    lons = [p.lon for p in cluster]
    centroid_lat = sum(lats) / len(lats)
    centroid_lon = sum(lons) / len(lons)
    radii = [
        haversine_km(centroid_lat, centroid_lon, p.lat, p.lon) for p in cluster
    ]
    radius_km = max(radii) if radii else 0.0

    timestamps = [(p.capture_ts or p.created_at) for p in cluster]
    started_at = min(timestamps)
    ended_at = max(timestamps)
    plant = cluster[0].plant or ""

    # R estimate from member capture days.
    g = generation_interval(disease)
    epoch_days = [_days_since_epoch(t) for t in timestamps]
    r_estimate = estimate_r_effective(epoch_days, g)

    hull = _convex_hull_geojson([(p.lat, p.lon) for p in cluster])

    ob = Outbreak(
        plant=plant,
        disease=disease,
        centroid_lat=centroid_lat,
        centroid_lon=centroid_lon,
        radius_km=radius_km,
        started_at=started_at,
        ended_at=ended_at,
        member_count=len(cluster),
        r_estimate=r_estimate,
        hull_geojson=hull,
    )
    session.add(ob)
    session.flush()  # populate ob.id

    for p in cluster:
        p.outbreak_id = ob.id
    return ob


def _days_since_epoch(t: datetime) -> float:
    return t.timestamp() / 86400.0
