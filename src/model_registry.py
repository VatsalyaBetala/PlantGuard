from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from src.grad_cam import generate_grad_cam
from src.model_adapters.cnn_adapter import CNNAdapter
from src.model_adapters.vit_adapter import ViTAdapter
from src.model_artifacts import artifact_dir, ensure_labels
from src.model_catalog import (
    DISEASE_LABELS,
    PLANT_CLASSES,
    disease_model_name,
    get_backend_name,
    plant_model_name,
)
from src.Disease_Classification_resnet50.src.disease_model import DiseaseClassifier
from src.Plant_Classification_resnet50.plant_classification_TL import PlantClassifierCNN


@dataclass(frozen=True)
class ModelSpec:
    adapter_class: type
    artifact_dir: str
    labels: List[str]
    model_class: Optional[type] = None
    grad_cam_context: Optional[dict] = None


_MODEL_REGISTRY: Dict[str, ModelSpec] = {}
_MODEL_CACHE: Dict[str, CNNAdapter] = {}


def _register_default_models() -> None:
    if _MODEL_REGISTRY:
        return

    backend = get_backend_name()
    plant_name = plant_model_name(backend)
    _MODEL_REGISTRY[plant_name] = ModelSpec(
        adapter_class=CNNAdapter,
        artifact_dir=str(artifact_dir(plant_name)),
        labels=PLANT_CLASSES,
        model_class=PlantClassifierCNN,
    )

    for plant, labels in DISEASE_LABELS.items():
        model_name = disease_model_name(plant, backend)
        _MODEL_REGISTRY[model_name] = ModelSpec(
            adapter_class=CNNAdapter,
            artifact_dir=str(artifact_dir(model_name)),
            labels=labels,
            model_class=DiseaseClassifier,
            grad_cam_context={\"plant_type\": plant},
        )

    vit_name = f"{backend.replace('cnn', 'vit')}_plant"
    _MODEL_REGISTRY[vit_name] = ModelSpec(
        adapter_class=ViTAdapter,
        artifact_dir=str(artifact_dir(vit_name)),
        labels=PLANT_CLASSES,
        model_class=None,
    )


def available_models() -> List[str]:
    _register_default_models()
    return sorted(_MODEL_REGISTRY.keys())


def get_model(model_name: str, artifact_dir_override: str | None = None, device: str | None = None):
    _register_default_models()
    ensure_labels()

    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}")

    spec = _MODEL_REGISTRY[model_name]
    artifact_dir_path = artifact_dir_override or spec.artifact_dir

    if spec.adapter_class is CNNAdapter:
        grad_cam_fn = generate_grad_cam if spec.grad_cam_context else None
        adapter = CNNAdapter(
            name=model_name,
            model_class=spec.model_class,
            labels=spec.labels,
            grad_cam_fn=grad_cam_fn,
            grad_cam_context=spec.grad_cam_context,
        )
    else:
        adapter = spec.adapter_class()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    adapter.load(artifact_dir_path, device)
    _MODEL_CACHE[model_name] = adapter
    return adapter
