import os
import gdown

from src.model_artifacts import LEGACY_MODEL_DIR, required_legacy_files, sync_legacy_to_artifacts


def check_and_download_models():
    """
    Checks if all required models exist in the specified directory.
    If missing, downloads the entire folder from Google Drive.
    """
    # Ensure the legacy model directory exists
    os.makedirs(LEGACY_MODEL_DIR, exist_ok=True)

    required_models = required_legacy_files()

    # Check which models are missing
    missing_models = [
        model for model in required_models if not os.path.exists(os.path.join(LEGACY_MODEL_DIR, model))
    ]

    if not missing_models:
        print("✅ All models are present.")
        sync_legacy_to_artifacts()
        return True  # No need to download

    print(f"⚠️ Missing models: {missing_models}")

    # Get Google Drive folder link from .env
    gdrive_link = "https://drive.google.com/drive/folders/1B0OFkmGCXMbl8Eokjw5MikacxaYQrzwM"
    if not gdrive_link:
        raise ValueError("❌ ERROR: `G_DRIVE_LINK` is not set in .env file.")

    print(f"📥 Downloading missing models from Google Drive: {gdrive_link}")

    try:
        # Download the entire folder
        gdown.download_folder(gdrive_link, output=str(LEGACY_MODEL_DIR), quiet=False, use_cookies=False)
        sync_legacy_to_artifacts()
        print("✅ All missing models downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to download models: {e}")
        return False
