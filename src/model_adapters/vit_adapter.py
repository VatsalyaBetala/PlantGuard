from __future__ import annotations

from typing import Any

from src.model_adapters.base import ModelAdapter


class ViTAdapter(ModelAdapter):
    def load(self, artifact_dir: str, device: str):
        raise FileNotFoundError(
            "ViT artifacts not found; train first. Expected: "
            f"{artifact_dir}/model.pt and labels.json"
        )

    def preprocess(self, image: Any):
        raise NotImplementedError("ViTAdapter preprocess not implemented yet.")

    def predict(self, tensor):
        raise NotImplementedError("ViTAdapter predict not implemented yet.")

    def postprocess(self, output) -> dict:
        raise NotImplementedError("ViTAdapter postprocess not implemented yet.")

    def explain(self, image: Any, pred: dict | None = None):
        return None
