# Model Artifacts

Standardized artifact layout for PlantGuard model backends.

```
artifacts/
  <model_name>/
    model.pt
    labels.json
    config.yaml (optional)
  shared/
    resnet50_weights.pth
    yolov8n_leaf.pt
```

Populate these folders with trained checkpoints. The application will auto-sync any
legacy `src/models` downloads into this layout on startup.
