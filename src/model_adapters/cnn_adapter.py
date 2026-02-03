from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, List, Optional

import torch

from src.data_preprocessing import preprocess_image_input
from src.model_adapters.base import ModelAdapter


class CNNAdapter(ModelAdapter):
    def __init__(
        self,
        name: str,
        model_class,
        labels: List[str] | None = None,
        preprocess_fn: Callable[[Any], torch.Tensor] | None = None,
        grad_cam_fn: Callable[..., Any] | None = None,
        grad_cam_context: dict | None = None,
    ) -> None:
        self.name = name
        self.model_class = model_class
        self.labels = labels or []
        self.preprocess_fn = preprocess_fn or preprocess_image_input
        self.grad_cam_fn = grad_cam_fn
        self.grad_cam_context = grad_cam_context or {}
        self.model = None
        self.device = None

    def load(self, artifact_dir: str, device: str):
        artifact_path = Path(artifact_dir)
        labels_path = artifact_path / "labels.json"
        model_path = artifact_path / "model.pt"

        if labels_path.exists():
            self.labels = json.loads(labels_path.read_text())
        if not self.labels:
            raise ValueError(f"No labels configured for model {self.name}.")
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")

        self.device = torch.device(device)
        model = self.model_class(len(self.labels))
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.to(self.device)
        model.eval()
        self.model = model
        return self

    def preprocess(self, image: Any):
        tensor = self.preprocess_fn(image)
        if self.device is not None:
            tensor = tensor.to(self.device)
        return tensor

    def predict(self, tensor):
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        with torch.no_grad():
            return self.model(tensor)

    def postprocess(self, output) -> dict:
        probabilities = torch.softmax(output, dim=1)
        topk_probs, topk_indices = torch.topk(probabilities, k=min(3, probabilities.shape[1]))
        topk_probs = topk_probs.squeeze(0).tolist()
        topk_indices = topk_indices.squeeze(0).tolist()
        topk = [
            {"label": self.labels[idx], "confidence": float(prob)}
            for idx, prob in zip(topk_indices, topk_probs)
        ]
        best = topk[0]
        return {
            "label": best["label"],
            "confidence": best["confidence"],
            "topk": topk,
        }

    def explain(self, image: Any, pred: Optional[dict] = None):
        if not self.grad_cam_fn:
            return None
        if isinstance(image, Path):
            image = str(image)
        if not isinstance(image, str):
            raise ValueError("Grad-CAM explain expects an image path for CNNAdapter.")
        return self.grad_cam_fn(image, **self.grad_cam_context)
