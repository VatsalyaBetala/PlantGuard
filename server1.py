from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from typing import List

app = FastAPI()

# CORS Middleware to allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

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

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    for file in files:
        file_location = f"uploads/{file.filename}"
        with open(file_location, "wb") as f:
            contents = await file.read()
            f.write(contents)
    return {"message": "Files successfully uploaded"}

@app.get("/images")
async def get_images():
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    files = os.listdir("uploads")
    return files

@app.delete("/delete-images")
async def delete_images():
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    files = os.listdir("uploads")
    for file in files:
        os.remove(f"uploads/{file}")
    return {"message": "All images deleted"}

if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    uvicorn.run(app, host="127.0.0.1", port=8000)
