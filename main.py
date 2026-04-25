import os
import uuid
import traceback
from typing import List
from pydantic import BaseModel
from pathlib import Path


import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
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
    classify_plant,
    classify_disease,
    explain_disease,
)
from plant_disease.db import init_db
from plant_disease.db.repo import (
    record_prediction,
    list_predictions,
    delete_prediction,
    delete_all_predictions,
)

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
os.makedirs("heatmaps", exist_ok=True)
_wipe_dir("uploads")
_wipe_dir("heatmaps")

# Serve static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/heatmaps", StaticFiles(directory="heatmaps"), name="heatmaps")

@app.on_event("startup")
async def startup_event():
    # Initialize the SQLite predictions DB (idempotent; creates the table on first run).
    init_db()
    # Check & download missing models
    check_and_download_models()

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


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    predictions = []
    for file in files:
        try:
            # Validate file type
            allowed_extensions = {"png", "jpg", "jpeg"}
            file_ext = file.filename.split(".")[-1].lower()
            if file_ext not in allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid file type. Only PNG, JPG, and JPEG are allowed."
                )

            # Generate unique filename
            unique_name = f"{uuid.uuid4()}.{file_ext}"
            file_location = f"uploads/{unique_name}"

            # Save the uploaded file
            with open(file_location, "wb") as f_out:
                contents = await file.read()
                f_out.write(contents)

            print(f"File {file.filename} saved as {unique_name}")

            # 1) Detect and crop the leaf
            cropped_leaf_path = detect_leaf(file_location)
            if cropped_leaf_path is None:
                plant_prediction = "No Leaf Detected"
                disease_prediction = "Unknown"
                heatmap_path = None
            else:
                # 2) Classify plant
                plant_prediction = classify_plant(cropped_leaf_path)
                print(f"Plant prediction: {plant_prediction}")

                # 3) Classify disease
                disease_prediction = classify_disease(cropped_leaf_path, plant_prediction)
                print(f"Disease prediction: {disease_prediction}")

                # 4) Generate Grad-CAM heatmap
                heatmap_path = explain_disease(cropped_leaf_path, plant_prediction)
            # Persist to SQLite. Confidence/top3 fields stay NULL until the
            # inference pipeline starts surfacing them (next feature).
            pred = record_prediction(
                filename=unique_name,
                original_name=file.filename or "",
                plant=plant_prediction,
                disease=disease_prediction,
                heatmap=os.path.basename(heatmap_path) if heatmap_path else None,
                source="upload",
            )
            predictions.append(pred)

        except Exception as e:
            print(f"Error during upload processing: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

    return {"message": "Files successfully uploaded", "predictions": predictions}


@app.get("/images")
async def get_images():
    """Return every persisted prediction. Newest first.

    The DB is the source of truth — files orphaned on disk (e.g. left over
    from a crash) are intentionally not surfaced.
    """
    return list_predictions()


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
        from plant_disease.db.repo import get_prediction
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


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
