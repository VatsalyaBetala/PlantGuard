import json
import os
import shutil
from pathlib import Path
from typing import Iterable, List

from src.model_catalog import (
    DISEASE_LABELS,
    LEGACY_DISEASE_MODEL_FILENAMES,
    LEGACY_PLANT_MODEL_FILENAME,
    LEGACY_RESNET_WEIGHTS,
    LEGACY_YOLO_WEIGHTS,
    PLANT_CLASSES,
    disease_model_name,
    plant_model_name,
)

ARTIFACTS_DIR = Path(os.getenv("MODEL_ARTIFACTS_DIR", "artifacts"))
LEGACY_MODEL_DIR = Path(os.getenv("LEGACY_MODEL_DIR", "src/models"))


def artifact_dir(model_name: str) -> Path:
    return ARTIFACTS_DIR / model_name


def model_file(model_name: str) -> Path:
    return artifact_dir(model_name) / "model.pt"


def labels_file(model_name: str) -> Path:
    return artifact_dir(model_name) / "labels.json"


def shared_file(filename: str) -> Path:
    return ARTIFACTS_DIR / "shared" / filename


def ensure_artifact_dirs(model_names: Iterable[str]) -> None:
    for model_name in model_names:
        artifact_dir(model_name).mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "shared").mkdir(parents=True, exist_ok=True)


def write_labels(model_name: str, labels: List[str]) -> None:
    label_path = labels_file(model_name)
    if label_path.exists():
        return
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(json.dumps(labels, indent=2))


def ensure_labels() -> None:
    write_labels(plant_model_name(), PLANT_CLASSES)
    for plant, labels in DISEASE_LABELS.items():
        write_labels(disease_model_name(plant), labels)


def legacy_model_path(filename: str) -> Path:
    return LEGACY_MODEL_DIR / filename


def resolve_shared_file(filename: str) -> Path:
    primary = shared_file(filename)
    legacy = legacy_model_path(filename)
    if primary.exists():
        return primary
    return legacy


def sync_legacy_to_artifacts() -> None:
    ensure_artifact_dirs([plant_model_name(), *[disease_model_name(p) for p in DISEASE_LABELS]])
    ensure_labels()

    legacy_resnet = legacy_model_path(LEGACY_RESNET_WEIGHTS)
    if legacy_resnet.exists() and not shared_file(LEGACY_RESNET_WEIGHTS).exists():
        shutil.copy2(legacy_resnet, shared_file(LEGACY_RESNET_WEIGHTS))

    legacy_yolo = legacy_model_path(LEGACY_YOLO_WEIGHTS)
    if legacy_yolo.exists() and not shared_file(LEGACY_YOLO_WEIGHTS).exists():
        shutil.copy2(legacy_yolo, shared_file(LEGACY_YOLO_WEIGHTS))

    plant_src = legacy_model_path(LEGACY_PLANT_MODEL_FILENAME)
    plant_dest = model_file(plant_model_name())
    if plant_src.exists() and not plant_dest.exists():
        shutil.copy2(plant_src, plant_dest)

    for plant, filename in LEGACY_DISEASE_MODEL_FILENAMES.items():
        legacy_path = legacy_model_path(filename)
        dest_path = model_file(disease_model_name(plant))
        if legacy_path.exists() and not dest_path.exists():
            shutil.copy2(legacy_path, dest_path)


def required_legacy_files() -> List[str]:
    required = [
        LEGACY_PLANT_MODEL_FILENAME,
        LEGACY_RESNET_WEIGHTS,
        LEGACY_YOLO_WEIGHTS,
    ]
    required.extend(LEGACY_DISEASE_MODEL_FILENAMES.values())
    return required


def resolve_model_path(model_name: str, legacy_filename: str | None = None) -> Path:
    candidate = model_file(model_name)
    if candidate.exists():
        return candidate
    if legacy_filename:
        legacy = legacy_model_path(legacy_filename)
        if legacy.exists():
            return legacy
    return candidate
