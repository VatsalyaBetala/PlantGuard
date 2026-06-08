import os
import uuid
import traceback
import json
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
from pathlib import Path


import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
from dotenv import load_dotenv
from openai import OpenAI
import openai
import logging

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Local imports
from plant_disease.utils import check_and_download_models
from plant_disease.inference import (
    detect_leaf,
    classify_plant_with_details,
    classify_disease_with_details,
    explain_disease,
)
from plant_disease.agent import enhance_triage_with_openai, triage_prediction
from plant_disease.clustering import recompute_outbreaks
from plant_disease.db import init_db
from plant_disease.db.repo import (
    get_outbreak,
    get_prediction,
    list_all_pressure,
    list_geo_predictions,
    list_outbreaks,
    list_outbreaks_near,
    list_predictions,
    list_pressure_for_grid,
    record_prediction,
    delete_prediction,
    delete_all_predictions,
    timeline_buckets,
)
from plant_disease.geo import parse_exif_gps
from plant_disease.pressure_jobs import compute_all_pressure

class DiagnosisRequest(BaseModel):
    plant: str
    disease: str
    
logger = logging.getLogger("uvicorn.error")   
   
# Initialize FastAPI
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # set to specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _wipe_dir(path: str) -> None:
    """Remove every file directly inside `path`, leaving the directory itself."""
    if not os.path.isdir(path):
        return
    for entry in os.listdir(path):
        full = os.path.join(path, entry)
        try:
            if os.path.isfile(full) or os.path.islink(full):
                os.remove(full)
        except OSError as exc:
            logger.warning("Could not remove %s during startup wipe: %s", full, exc)


# Ensure uploads / heatmaps directories exist, then wipe them so we start fresh.
# Stale files on disk would not have matching DB rows after the schema change.
os.makedirs("uploads", exist_ok=True)
os.makedirs("uploads/_crops", exist_ok=True)
os.makedirs("heatmaps", exist_ok=True)
_wipe_dir("uploads")
_wipe_dir("uploads/_crops")
_wipe_dir("heatmaps")
# Legacy single-slot crop file from earlier versions — stale, drop it.
if os.path.exists("temp_leaf.jpg"):
    try:
        os.remove("temp_leaf.jpg")
    except OSError:
        pass

# Serve static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/heatmaps", StaticFiles(directory="heatmaps"), name="heatmaps")

_scheduler = None


@app.on_event("startup")
async def startup_event():
    global _scheduler
    # Initialize the SQLite predictions DB (idempotent; creates the table on first run).
    init_db()
    # Check & download missing models
    check_and_download_models()

    # Background jobs: cluster recompute + pressure refresh.
    if os.getenv("PLANTGUARD_DISABLE_SCHEDULER") != "1":
        try:
            from apscheduler.schedulers.background import BackgroundScheduler

            sched = BackgroundScheduler(daemon=True)
            sched.add_job(_safe_recompute_outbreaks, "interval", minutes=15, id="cluster")
            sched.add_job(_safe_refresh_pressure, "interval", hours=1, id="pressure")
            sched.start()
            _scheduler = sched
            logger.info("Outbreak scheduler started (cluster=15m, pressure=1h)")
        except Exception as exc:
            logger.warning("Scheduler failed to start: %s", exc)


def _safe_recompute_outbreaks() -> None:
    try:
        result = recompute_outbreaks()
        logger.info("Auto-recluster: %s", result)
    except Exception:
        logger.exception("Auto-recluster failed")


def _safe_refresh_pressure() -> None:
    try:
        result = compute_all_pressure()
        logger.info("Auto-pressure: %s", result)
    except Exception:
        logger.exception("Auto-pressure failed")


@app.on_event("shutdown")
async def shutdown_event():
    global _scheduler
    if _scheduler is not None:
        try:
            _scheduler.shutdown(wait=False)
        except Exception:
            pass

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return FileResponse("views/index.html")

@app.get("/live-view", response_class=HTMLResponse)
async def live_view():
    return FileResponse("views/live-view.html")

@app.get("/upload-images", response_class=HTMLResponse)
async def image_view():
    return FileResponse("views/upload-images.html")

@app.get("/view-images", response_class=HTMLResponse)
async def view_images():
    return FileResponse("views/view-images.html")

@app.get("/record-video", response_class=HTMLResponse)
async def record_video():
    return FileResponse("views/record-video.html")

@app.get("/about", response_class=HTMLResponse)
async def about_page():
    return FileResponse("views/about.html")

@app.get("/doc", response_class=HTMLResponse)
async def docs_page():
    return FileResponse("views/docs.html")

@app.get("/static/docs/{filename}")
async def get_document(filename: str):
    """Serve individual document files with proper headers"""
    file_path = Path(f"static/docs/{filename}")
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found")
    
    try:
        file_path.resolve().relative_to(Path("static/docs").resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    
    media_type = "application/octet-stream"
    if filename.lower().endswith('.pdf'):
        media_type = "application/pdf"
    elif filename.lower().endswith('.docx'):
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif filename.lower().endswith('.doc'):
        media_type = "application/msword"
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type=media_type,
        headers={
            "Content-Disposition": f"inline; filename={filename}",
            "Cache-Control": "public, max-age=3600"
        }
    )

@app.get("/contact", response_class=HTMLResponse)
async def contact_page():
    return FileResponse("views/contact.html")

@app.get("/faq", response_class=HTMLResponse)
async def faq_page():
    return FileResponse("views/faq.html")


@app.get("/outbreaks", response_class=HTMLResponse)
async def outbreaks_page():
    return FileResponse("views/outbreaks.html")


@app.post("/upload")
async def upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
    geo_accuracy_m: Optional[float] = Form(None),
    geo_source: Optional[str] = Form(None),
    capture_ts: Optional[str] = Form(None),
):
    predictions = []
    used_geo = False
    for file in files:
        try:
            allowed_extensions = {"png", "jpg", "jpeg"}
            file_ext = file.filename.split(".")[-1].lower()
            if file_ext not in allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid file type. Only PNG, JPG, and JPEG are allowed."
                )

            unique_name = f"{uuid.uuid4()}.{file_ext}"
            file_location = f"uploads/{unique_name}"

            with open(file_location, "wb") as f_out:
                contents = await file.read()
                f_out.write(contents)

            print(f"File {file.filename} saved as {unique_name}")

            # ---- Geo resolution: form-supplied wins, EXIF as fallback ----
            row_lat = lat
            row_lon = lon
            row_acc = geo_accuracy_m
            row_src = geo_source
            row_capture: Optional[datetime] = None
            if capture_ts:
                try:
                    row_capture = datetime.fromisoformat(capture_ts.replace("Z", "+00:00"))
                    if row_capture.tzinfo is not None:
                        row_capture = row_capture.replace(tzinfo=None)
                except ValueError:
                    row_capture = None

            if row_lat is None or row_lon is None:
                exif = parse_exif_gps(file_location)
                if exif is not None:
                    row_lat = exif["lat"]
                    row_lon = exif["lon"]
                    row_src = row_src or "exif"
                    if row_capture is None and exif.get("capture_ts"):
                        row_capture = exif["capture_ts"]

            # 1) Detect and crop the leaf
            cropped_leaf_path = detect_leaf(file_location)
            if cropped_leaf_path is None:
                plant_prediction = "No Leaf Detected"
                disease_prediction = "Unknown"
                plant_conf = None
                disease_conf = None
                top3_json = None
                heatmap_path = None
            else:
                plant_result = classify_plant_with_details(cropped_leaf_path)
                plant_prediction = plant_result["label"]
                plant_conf = plant_result["confidence"]
                print(f"Plant prediction: {plant_prediction} ({plant_conf:.2f})")

                disease_result = classify_disease_with_details(cropped_leaf_path, plant_prediction)
                disease_prediction = disease_result["label"]
                disease_conf = disease_result["confidence"]
                if disease_conf is None:
                    print(f"Disease prediction: {disease_prediction}")
                else:
                    print(f"Disease prediction: {disease_prediction} ({disease_conf:.2f})")

                top3_json = json.dumps(
                    {
                        "plant": plant_result.get("topk", []),
                        "disease": disease_result.get("topk", []),
                    }
                )

                heatmap_path = explain_disease(cropped_leaf_path, plant_prediction)

            pred = record_prediction(
                filename=unique_name,
                original_name=file.filename or "",
                plant=plant_prediction,
                disease=disease_prediction,
                heatmap=os.path.basename(heatmap_path) if heatmap_path else None,
                plant_conf=plant_conf,
                disease_conf=disease_conf,
                top3_json=top3_json,
                source="upload",
                lat=row_lat,
                lon=row_lon,
                geo_accuracy_m=row_acc,
                geo_source=row_src,
                capture_ts=row_capture,
            )
            predictions.append(pred)
            if row_lat is not None and row_lon is not None:
                used_geo = True

        except Exception as e:
            print(f"Error during upload processing: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

    # If any upload was geo-tagged, kick a re-cluster + pressure refresh in the background.
    if used_geo:
        background_tasks.add_task(_safe_recompute_outbreaks)
        background_tasks.add_task(_safe_refresh_pressure)

    return {"message": "Files successfully uploaded", "predictions": predictions}


@app.get("/images")
async def get_images():
    """Return every persisted prediction. Newest first.

    The DB is the source of truth — files orphaned on disk (e.g. left over
    from a crash) are intentionally not surfaced.
    """
    return list_predictions()


@app.get("/agent/triage/{filename}")
async def triage_image(filename: str):
    """Run triage for a stored prediction.

    Uses local deterministic triage first, then enhances it with OpenAI when
    OPENAI_API_KEY is configured. If OpenAI fails, local triage is returned.
    """
    row = get_prediction(filename)
    if row is None:
        raise HTTPException(status_code=404, detail="Image prediction not found")

    local_triage = triage_prediction(row)
    if not openai.api_key:
        return {
            **local_triage,
            "openai_status": "skipped_missing_api_key",
        }

    try:
        client = OpenAI(api_key=openai.api_key)
        return {
            **enhance_triage_with_openai(
                row,
                local_triage,
                client,
                model=os.getenv("OPENAI_TRIAGE_MODEL", "gpt-4o"),
                history_fn=list_predictions,
            ),
            "openai_status": "used",
        }
    except Exception as exc:
        logger.exception("OpenAI triage enhancement failed")
        return {
            **local_triage,
            "openai_status": "fallback_after_error",
            "openai_error": str(exc),
        }


@app.post("/explain-diagnosis")
async def explain_diagnosis(req: DiagnosisRequest):
    plant, disease = req.plant, req.disease

    if not openai.api_key:
        logging.getLogger("uvicorn.error").error("OPENAI_API_KEY not set")
        raise HTTPException(500, "Server misconfigured: missing API key")

    prompt = (f"""
You are a plant pathology expert. Provide a detailed explanation for the following diagnosis:

- **Plant**: {plant}
- **Disease**: {disease}

Break your response into the following sections:

1. Overview: A short description of the plant and its importance.
2. About the Disease: What is this disease, and how does it affect the plant?
3. Causes: What causes this disease (e.g., bacteria, fungi, environmental factors)?
4. Symptoms: What are the common visual signs farmers should look for?
5. Impact: How does this disease affect plant health or yield?
6. Treatment: What are effective remedies or treatments?
7. Prevention: How can farmers prevent this disease in the future?

Write in simple, farmer-friendly language with bullet points where appropriate.
"""
    )

    try:
        resp = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        explanation = resp.choices[0].message.content
        return {"plant": plant, "disease": disease, "explanation": explanation}

    except Exception as e:
        logging.getLogger("uvicorn.error").exception("LLM call failed")
        raise HTTPException(500, f"LLM error: {e}")

@app.delete("/delete-images")
async def delete_images():
    try:
        rows = list_predictions()
        for row in rows:
            fn = row["filename"]
            file_path = os.path.join("uploads", fn)
            if os.path.exists(file_path):
                os.remove(file_path)
            heatmap_name = row.get("heatmap")
            if heatmap_name:
                heatmap_path = os.path.join("heatmaps", heatmap_name)
                if os.path.exists(heatmap_path):
                    os.remove(heatmap_path)
        deleted = delete_all_predictions()
        return {"message": "All images deleted", "deleted": deleted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete images: {str(e)}")


@app.delete("/images/{filename}")
async def delete_image(filename: str):
    """Delete a single uploaded image and its prediction row."""
    try:
        file_path = os.path.join("uploads", filename)
        existed_on_disk = os.path.exists(file_path)
        if existed_on_disk:
            os.remove(file_path)

        # Heatmap is named the same as the cropped leaf, not the upload filename,
        # so look it up via the DB row before deleting it.
        row = get_prediction(filename)
        if row and row.get("heatmap"):
            heatmap_path = os.path.join("heatmaps", row["heatmap"])
            if os.path.exists(heatmap_path):
                os.remove(heatmap_path)

        deleted = delete_prediction(filename)
        if not existed_on_disk and not deleted:
            raise HTTPException(status_code=404, detail="Image not found")
        return {"message": f"{filename} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete image: {e}")


# ---------------------------------------------------------------------------
# Map / outbreak APIs
# ---------------------------------------------------------------------------

def _predictions_to_geojson(rows: list[dict]) -> dict:
    features = []
    for r in rows:
        if r.get("lat") is None or r.get("lon") is None:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
                "properties": {
                    "filename": r.get("filename"),
                    "plant": r.get("plant"),
                    "disease": r.get("disease"),
                    "outbreak_id": r.get("outbreak_id"),
                    "capture_ts": r.get("capture_ts"),
                    "created_at": r.get("created_at"),
                    "geo_source": r.get("geo_source"),
                    "disease_conf": r.get("disease_conf"),
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _outbreaks_to_geojson(rows: list[dict]) -> dict:
    features = []
    for ob in rows:
        hull = ob.get("hull_geojson")
        geometry = None
        if hull:
            try:
                geometry = json.loads(hull)
            except json.JSONDecodeError:
                geometry = None
        if geometry is None:
            c = ob.get("centroid") or {}
            if c.get("lat") is None or c.get("lon") is None:
                continue
            geometry = {"type": "Point", "coordinates": [c["lon"], c["lat"]]}
        features.append(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "id": ob["id"],
                    "plant": ob["plant"],
                    "disease": ob["disease"],
                    "centroid": ob.get("centroid"),
                    "radius_km": ob.get("radius_km"),
                    "started_at": ob.get("started_at"),
                    "ended_at": ob.get("ended_at"),
                    "member_count": ob.get("member_count"),
                    "r_estimate": ob.get("r_estimate"),
                    "pressure_score": ob.get("pressure_score"),
                    "pressure_model": ob.get("pressure_model"),
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


@app.get("/map/predictions")
async def map_predictions(
    plant: Optional[str] = Query(None),
    disease: Optional[str] = Query(None),
    since_days: int = Query(90, ge=1, le=365),
    jitter: bool = Query(True),
):
    rows = list_geo_predictions(since_days=since_days, plant=plant, disease=disease, jitter=jitter)
    return JSONResponse(_predictions_to_geojson(rows))


@app.get("/map/outbreaks")
async def map_outbreaks(active_only: bool = Query(True)):
    rows = list_outbreaks(active_only=active_only)
    return JSONResponse(_outbreaks_to_geojson(rows))


@app.get("/map/outbreaks/{outbreak_id}")
async def map_outbreak_detail(outbreak_id: int):
    ob = get_outbreak(outbreak_id)
    if ob is None:
        raise HTTPException(status_code=404, detail="Outbreak not found")
    return ob


@app.get("/map/pressure")
async def map_pressure(
    disease: Optional[str] = Query(None),
    day: Optional[str] = Query(None),
):
    rows = list_all_pressure(day=day, disease=disease)
    return {"items": rows, "count": len(rows)}


@app.get("/map/pressure/local")
async def map_pressure_local(
    lat: float = Query(...),
    lon: float = Query(...),
    disease: Optional[str] = Query(None),
):
    rows = list_pressure_for_grid(lat, lon, disease=disease)
    nearby = list_outbreaks_near(lat, lon, radius_km=15.0, days=14)
    return {"pressure": rows, "nearby_outbreaks": nearby}


@app.get("/map/timeline")
async def map_timeline(since_days: int = Query(90, ge=1, le=365)):
    return {"buckets": timeline_buckets(since_days=since_days)}


@app.get("/map/diseases")
async def map_diseases():
    """Distinct (plant, disease) pairs present in geo-tagged predictions."""
    rows = list_geo_predictions(since_days=365, jitter=False)
    pairs: dict[str, set[str]] = {}
    for r in rows:
        pl = r.get("plant") or "Unknown"
        ds = r.get("disease") or "Unknown"
        pairs.setdefault(pl, set()).add(ds)
    return {plant: sorted(d) for plant, d in pairs.items()}


@app.post("/map/recompute")
async def map_recompute(background_tasks: BackgroundTasks):
    background_tasks.add_task(_safe_recompute_outbreaks)
    background_tasks.add_task(_safe_refresh_pressure)
    return {"message": "Recompute scheduled"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
