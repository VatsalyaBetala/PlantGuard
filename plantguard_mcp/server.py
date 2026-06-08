from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
from fastmcp import FastMCP

# Initialize the server 
mcp = FastMCP("PlantGuard")

PLANTGUARD_API_BASE = os.getenv("PLANTGUARD_API_BASE", "http://127.0.0.1:8000").rstrip("/")

_ALLOWED_EXTENSIONS = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}

# Helper function to construct artifact URLs based on the backend's expected paths.
def _artifact_url(kind: str, filename: str | None) -> str | None:
    if not filename:
        return None
    return f"{PLANTGUARD_API_BASE}/{kind}/{filename}"

def _diagnosis_status(plant: str | None, disease: str | None) -> str:
    if plant in {"No Leaf Detected", "Unknown"} or disease in {None, "", "Unknown"}:
        return "needs_better_image"
    return "diagnosed"

def _reasoning_notes(status: str, heatmap: str | None) -> list[str]:
    if status == "needs_better_image":
        return [
            "PlantGuard could not produce a complete plant and disease diagnosis from this image.",
            "Ask for a clearer close-up image of a single leaf in natural light.",
        ]

    notes = ["PlantGuard detected a leaf and completed plant and disease classification."]
    if heatmap:
        notes.append("A Grad-CAM heatmap is available for visual review.")
    else:
        notes.append("No Grad-CAM heatmap was returned for this prediction.")
    return notes


def _normalize_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    plant_label = prediction.get("plant")
    disease_label = prediction.get("disease")
    heatmap = prediction.get("heatmap")
    filename = prediction.get("filename")
    status = _diagnosis_status(plant_label, disease_label)

    return {
        "status": status,
        "filename": filename,
        "plant": {
            "label": plant_label,
            "confidence": prediction.get("plant_conf"),
        },
        "disease": {
            "label": disease_label,
            "confidence": prediction.get("disease_conf"),
        },
        "artifacts": {
            "image_url": _artifact_url("uploads", filename),
            "heatmap_url": _artifact_url("heatmaps", heatmap),
        },
        "reasoning_notes": _reasoning_notes(status, heatmap),
    }

@mcp.tool()
async def diagnose_plant_image(
    image_path: str, # Path to a local image file
    lat: float | None = None, # Optional latitude
    lon: float | None = None, # Optional longitude
) -> dict[str, Any]: # Returns JSON-serializable dict
    """Diagnose one plant leaf image using the existing PlantGuard FastAPI backend."""

    path = Path(image_path).expanduser().resolve() # Converts into an absolute path 

    if not path.exists() or not path.is_file(): # Check for existence
        raise ValueError("image_path must point to an existing image file")

    media_type = _ALLOWED_EXTENSIONS.get(path.suffix.lower()) # Determine media type based on file extension 
    if media_type is None:
        raise ValueError("Only JPG, JPEG, and PNG images are supported")

# Prepare form data for the POST request
    data: dict[str, str] = {}
    if lat is not None:
        data["lat"] = str(lat)
    if lon is not None:
        data["lon"] = str(lon)

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            with path.open("rb") as image_file:
                response = await client.post(
                    f"{PLANTGUARD_API_BASE}/upload",
                    files={"files": (path.name, image_file, media_type)},
                    data=data,
                )
            response.raise_for_status()
    except httpx.ConnectError as exc:
        raise RuntimeError(
            f"PlantGuard backend is not reachable at {PLANTGUARD_API_BASE}. "
            "Start the FastAPI app first with `python main.py`."
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"PlantGuard backend returned HTTP {exc.response.status_code}: {exc.response.text}"
        ) from exc

    payload = response.json()
    predictions = payload.get("predictions") or []
    if not predictions:
        raise RuntimeError("PlantGuard backend returned no predictions")

    return _normalize_prediction(predictions[0])

if __name__ == "__main__":
    mcp.run()
