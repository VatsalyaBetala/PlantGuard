"""Local persistence layer for PlantGuard predictions.

Public API:
    init_db()         -> create tables if missing, return Engine
    session_scope()   -> contextmanager yielding a SQLAlchemy Session
    DATABASE_URL      -> the resolved SQLAlchemy URL

Designed so swapping SQLite for Postgres later is a single env var:
    DATABASE_URL=postgresql+psycopg://user:pass@host/plantguard
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from plant_disease.db.models import Base


def _default_sqlite_url() -> str:
    data_dir = Path(os.getenv("PLANTGUARD_DATA_DIR", "data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(data_dir / 'plantguard.db').as_posix()}"


DATABASE_URL: str = os.getenv("DATABASE_URL") or _default_sqlite_url()

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def _build_engine() -> Engine:
    is_sqlite = DATABASE_URL.startswith("sqlite")
    connect_args = {"check_same_thread": False} if is_sqlite else {}
    return create_engine(DATABASE_URL, connect_args=connect_args, future=True)


def init_db() -> Engine:
    """Create tables if they don't exist. Idempotent. Safe to call on every startup."""
    global _engine, _SessionLocal
    if _engine is None:
        _engine = _build_engine()
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False, future=True)
    Base.metadata.create_all(_engine)
    return _engine


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commit on success, rollback on exception, always close."""
    if _SessionLocal is None:
        init_db()
    assert _SessionLocal is not None
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
