"""Thin repository for predictions.

main.py talks only to these functions — never to SQLAlchemy directly.
That keeps the swap to Postgres later mechanical.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import delete, select

from plant_disease.db import session_scope
from plant_disease.db.models import Prediction


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
