import os
import gdown

MODEL_DIR = "src/models"
REQUIRED_MODELS = [
    "Apple_Disease_Classification.pth",
    "Grape_Disease_Classification.pth",
    "Plant_Classification.pth",
    "Tomato_Disease_Classification.pth",
    "Corn_(maize)_Disease_Classification.pth",
    "Pepper_bell_Disease_Classification.pth",
    "Potato_Disease_Classification.pth",
    "resnet50_weights.pth",
    "yolov8n_leaf.pt",
    "yolov8n_leaf.onnx"
]
def check_and_download_models():
    """
    Checks if all required models exist in the specified directory.
    If missing, downloads the entire folder from Google Drive.
    """
    # Ensure the model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Check which models are missing
    missing_models = [model for model in REQUIRED_MODELS if not os.path.exists(os.path.join(MODEL_DIR, model))]

    if not missing_models:
        print("✅ All models are present.")
        return True  # No need to download

    print(f"⚠️ Missing models: {missing_models}")

    # Get Google Drive folder link from .env
    gdrive_link = "https://drive.google.com/drive/folders/1B0OFkmGCXMbl8Eokjw5MikacxaYQrzwM"
    if not gdrive_link:
        raise ValueError("❌ ERROR: `G_DRIVE_LINK` is not set in .env file.")

    print(f"📥 Downloading missing models from Google Drive: {gdrive_link}")

    try:
        # Download the entire folder
        gdown.download_folder(gdrive_link, output=MODEL_DIR, quiet=False, use_cookies=False)
        print("✅ All missing models downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to download models: {e}")
        return False