import gdown
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def download_gdrive_folder(output_path="models/"):
    gdrive_link = os.getenv("G_DRIVE_LINK")
    if not gdrive_link:
        raise ValueError("G_DRIVE_LINK is not set in .env file")

    # Ensure the output directory exists
    output_path = os.path.abspath(output_path)  # Convert to absolute path
    os.makedirs(output_path, exist_ok=True)  # Create folder if it doesn't exist

    # Download the folder
    gdown.download_folder(gdrive_link, output=output_path, quiet=False, use_cookies=False)
