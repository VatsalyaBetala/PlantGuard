"""SQLAlchemy ORM models for PlantGuard."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Index, Integer, String, Text, func
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

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp()
    )

    __table_args__ = (
        Index("idx_predictions_created_at", "created_at"),
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
            "backend": self.backend,
            "source": self.source,
            "uploadDate": self.created_at.isoformat() if self.created_at else None,
            "originalName": self.original_name,
        }
