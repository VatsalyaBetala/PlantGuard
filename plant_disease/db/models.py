"""SQLAlchemy ORM models for PlantGuard."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # uuid filename written under uploads/
    filename: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    # what the user originally named the file
    original_name: Mapped[str] = mapped_column(String(512), nullable=False, default="")

    plant: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    disease: Mapped[str] = mapped_column(String(128), nullable=False, default="")

    # Future-proofing for the next features (per-prediction confidence, top-3, ViT vs CNN).
    # Nullable today; the inference pipeline can fill them later without a schema change.
    plant_conf: Mapped[float | None] = mapped_column(Float, nullable=True)
    disease_conf: Mapped[float | None] = mapped_column(Float, nullable=True)
    top3_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # basename of the heatmap PNG inside heatmaps/, or NULL when leaf detection failed
    heatmap: Mapped[str | None] = mapped_column(String(255), nullable=True)

    backend: Mapped[str] = mapped_column(String(64), nullable=False, default="cnn_resnet50")
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="upload")

    # Geo. Nullable: legacy rows + opted-out users.
    lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    geo_accuracy_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    geo_source: Mapped[str | None] = mapped_column(String(16), nullable=True)  # exif|browser|manual
    capture_ts: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Linked outbreak after clustering. Nullable until clustered.
    outbreak_id: Mapped[int | None] = mapped_column(
        ForeignKey("outbreaks.id", ondelete="SET NULL"), nullable=True, index=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp()
    )

    __table_args__ = (
        Index("idx_predictions_created_at", "created_at"),
        Index("idx_predictions_geo", "lat", "lon"),
        Index("idx_predictions_disease", "disease"),
    )

    def to_dict(self) -> dict:
        """Shape kept compatible with the existing /images JSON response."""
        return {
            "filename": self.filename,
            "plant": self.plant,
            "disease": self.disease,
            "heatmap": self.heatmap,
            # extras the frontend can ignore for now; needed by upcoming features
            "plant_conf": self.plant_conf,
            "disease_conf": self.disease_conf,
            "top3_json": self.top3_json,
            "backend": self.backend,
            "source": self.source,
            "uploadDate": self.created_at.isoformat() if self.created_at else None,
            "originalName": self.original_name,
            "lat": self.lat,
            "lon": self.lon,
            "geo_accuracy_m": self.geo_accuracy_m,
            "geo_source": self.geo_source,
            "capture_ts": self.capture_ts.isoformat() if self.capture_ts else None,
            "outbreak_id": self.outbreak_id,
        }


class WeatherSnap(Base):
    """Cached Open-Meteo lookup for one grid cell + day window."""
    __tablename__ = "weather_snaps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lat_grid: Mapped[float] = mapped_column(Float, nullable=False)  # rounded to grid step
    lon_grid: Mapped[float] = mapped_column(Float, nullable=False)
    window_kind: Mapped[str] = mapped_column(String(16), nullable=False)  # past|forecast
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp()
    )
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        Index("idx_weather_grid", "lat_grid", "lon_grid", "window_kind"),
    )


class Outbreak(Base):
    """A spatiotemporal cluster of diseased predictions of one disease."""
    __tablename__ = "outbreaks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    plant: Mapped[str] = mapped_column(String(64), nullable=False)
    disease: Mapped[str] = mapped_column(String(128), nullable=False)
    centroid_lat: Mapped[float] = mapped_column(Float, nullable=False)
    centroid_lon: Mapped[float] = mapped_column(Float, nullable=False)
    radius_km: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    ended_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    member_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    r_estimate: Mapped[float | None] = mapped_column(Float, nullable=True)
    hull_geojson: Mapped[str | None] = mapped_column(Text, nullable=True)
    pressure_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    pressure_model: Mapped[str | None] = mapped_column(String(32), nullable=True)
    refreshed_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp()
    )

    __table_args__ = (
        Index("idx_outbreaks_disease", "disease"),
        Index("idx_outbreaks_window", "started_at", "ended_at"),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "plant": self.plant,
            "disease": self.disease,
            "centroid": {"lat": self.centroid_lat, "lon": self.centroid_lon},
            "radius_km": self.radius_km,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat(),
            "member_count": self.member_count,
            "r_estimate": self.r_estimate,
            "hull_geojson": self.hull_geojson,
            "pressure_score": self.pressure_score,
            "pressure_model": self.pressure_model,
            "refreshed_at": self.refreshed_at.isoformat(),
        }


class PressureScore(Base):
    """Per (grid, plant, disease, day) disease pressure forecast."""
    __tablename__ = "pressure_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lat_grid: Mapped[float] = mapped_column(Float, nullable=False)
    lon_grid: Mapped[float] = mapped_column(Float, nullable=False)
    plant: Mapped[str] = mapped_column(String(64), nullable=False)
    disease: Mapped[str] = mapped_column(String(128), nullable=False)
    day: Mapped[str] = mapped_column(String(10), nullable=False)  # YYYY-MM-DD
    score: Mapped[float] = mapped_column(Float, nullable=False)
    model: Mapped[str] = mapped_column(String(32), nullable=False)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp()
    )

    __table_args__ = (
        Index("idx_pressure_lookup", "lat_grid", "lon_grid", "disease", "day"),
    )

    def to_dict(self) -> dict:
        return {
            "lat": self.lat_grid,
            "lon": self.lon_grid,
            "plant": self.plant,
            "disease": self.disease,
            "day": self.day,
            "score": self.score,
            "model": self.model,
        }
