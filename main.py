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
from src.utils import check_and_download_models
from src.inference import (
    detect_leaf,
    classify_plant,
    classify_disease
)
from src.grad_cam import generate_grad_cam

class DiagnosisRequest(BaseModel):
    plant: str
    disease: str
    
logger = logging.getLogger("uvicorn.error")   
   
# Initialize FastAPI
app = FastAPI()

# Simple in-memory store (use a real database in production)
PREDICTIONS = {}

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # set to specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure uploads directory
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# Ensure heatmaps directory
if not os.path.exists("heatmaps"):
    os.makedirs("heatmaps")

# Serve static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/heatmaps", StaticFiles(directory="heatmaps"), name="heatmaps")

@app.on_event("startup")
async def startup_event():
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
                heatmap_path = generate_grad_cam(cropped_leaf_path, plant_prediction)
            # Store prediction
            pred = {
                "filename": unique_name,
                "plant": plant_prediction,
                "disease": disease_prediction,
                "heatmap": os.path.basename(heatmap_path) if heatmap_path else None,
            }
            predictions.append(pred)
            PREDICTIONS[unique_name] = pred

        except Exception as e:
            print(f"Error during upload processing: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

    return {"message": "Files successfully uploaded", "predictions": predictions}


@app.get("/images")
async def get_images():
    files = os.listdir("uploads")
    results = []
    for file in files:
        pred = PREDICTIONS.get(file, {"plant": "", "disease": "", "heatmap": None})
        results.append({
            "filename": file,
            "plant": pred.get("plant", ""),
            "disease": pred.get("disease", ""),
            "heatmap": pred.get("heatmap")
        })
    return results


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
        files = os.listdir("uploads")
        for file in files:
            os.remove(f"uploads/{file}")
            heatmap_path = os.path.join("heatmaps", file)
            if os.path.exists(heatmap_path):
                os.remove(heatmap_path)
            if file in PREDICTIONS:
                del PREDICTIONS[file]
        return {"message": "All images deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete images: {str(e)}")


@app.delete("/images/{filename}")
async def delete_image(filename: str):
    """Delete a single uploaded image and its cached prediction."""
    try:
        file_path = os.path.join("uploads", filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Image not found")

        os.remove(file_path)
        heatmap_path = os.path.join("heatmaps", filename)
        if os.path.exists(heatmap_path):
            os.remove(heatmap_path)
        if filename in PREDICTIONS:
            del PREDICTIONS[filename]
        return {"message": f"{filename} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete image: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)