from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ModelAdapter(ABC):
    @abstractmethod
    def load(self, artifact_dir: str, device: str):
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, image: Any):
        raise NotImplementedError

    @abstractmethod
    def predict(self, tensor):
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, output) -> dict:
        raise NotImplementedError

    @abstractmethod
    def explain(self, image: Any, pred: dict | None = None):
        raise NotImplementedError
