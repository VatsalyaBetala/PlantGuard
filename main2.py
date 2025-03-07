import os
import time
import threading
import queue
import cv2
import tempfile
import logging
import sys
from dotenv import load_dotenv
from src.LCD import LCD
from src.Button import Button  # Import the button class
from src.utils import check_and_download_models
from src.inference import detect_leaf, classify_plant, classify_disease

# Load environment variables
load_dotenv()

# Configure logging: log to file and console.
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

file_handler = logging.FileHandler('app.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Configuration parameters from environment variables (with defaults)
UPLOADS_FOLDER = os.getenv("UPLOADS_FOLDER", "uploads")
PHONE_CAM_URL = os.getenv("PHONE_CAM_URL", "http://10.10.105.29:4747/video")
BUTTON_PIN = int(os.getenv("BUTTON_PIN", 18))
QUEUE_MAXSIZE = int(os.getenv("QUEUE_MAXSIZE", 10))
COOLDOWN_PERIOD = int(os.getenv("COOLDOWN_PERIOD", 10))
FRAME_SKIP = int(os.getenv("FRAME_SKIP", 20))
LCD_TIMEOUT = int(os.getenv("LCD_TIMEOUT", 5))
CAMERA_MAX_RETRIES = int(os.getenv("CAMERA_MAX_RETRIES", 5))
CAMERA_RETRY_DELAY = int(os.getenv("CAMERA_RETRY_DELAY", 5))
CAMERA_MAX_CONSECUTIVE_FAILURES = int(os.getenv("CAMERA_MAX_CONSECUTIVE_FAILURES", 5))

# Ensure uploads folder exists
if not os.path.exists(UPLOADS_FOLDER):
    os.makedirs(UPLOADS_FOLDER)

# Uncomment if needed: check_and_download_models()

# Create a queue for inter-thread communication
image_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)

# Global variables to track detection times and results
last_detection_time = 0

# Initialize hardware interfaces
button = Button(button_pin=BUTTON_PIN)
lcd = LCD()
lcd.clear()

# Event to signal threads to stop
stop_event = threading.Event()

def run_inference(image_path):
    """
    Runs the inference pipeline: detect leaf, classify plant, classify disease.
    Returns a tuple of (plant, disease).
    """
    try:
        cropped_leaf_path = detect_leaf(image_path)
        if cropped_leaf_path is None:
            return "No Leaf Detected", "Unknown"
        plant = classify_plant(cropped_leaf_path)
        disease = classify_disease(cropped_leaf_path, plant)
        return plant, disease
    except Exception as e:
        logger.error(f"Error in inference: {e}")
        return "Error", "Error"

def capture_frames():
    """
    Capture frames from the camera and add them to the processing queue.
    Implements retry logic for the initial connection and reinitializes the camera
    if consecutive frame capture failures occur.
    If the queue is full, clears old frames so that only the latest frame is kept.
    """
    consecutive_failures = 0
    cap = None

    # Attempt to establish the camera connection with retries.
    for attempt in range(CAMERA_MAX_RETRIES):
        # Use cv2.VideoCapture(0) for a local USB camera
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            logger.info("Camera connection established.")
            break
        else:
            logger.error(f"Attempt {attempt+1}/{CAMERA_MAX_RETRIES}: Could not open camera. Retrying in {CAMERA_RETRY_DELAY} seconds.")
            time.sleep(CAMERA_RETRY_DELAY)

    if cap is None or not cap.isOpened():
        logger.error("Max retries reached. Could not open the camera. Exiting capture thread.")
        stop_event.set()
        return

    frame_counter = 0
    while not stop_event.is_set():
        if not button.check_button():
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            logger.warning("Frame capture failed. Attempting to reinitialize camera.")
            consecutive_failures += 1
            if consecutive_failures >= CAMERA_MAX_CONSECUTIVE_FAILURES:
                logger.error("Too many consecutive frame failures. Reinitializing camera connection.")
                cap.release()
                for attempt in range(CAMERA_MAX_RETRIES):
                    cap = cv2.VideoCapture(0)
                    if cap.isOpened():
                        logger.info("Camera reinitialized successfully.")
                        consecutive_failures = 0
                        break
                    else:
                        logger.error(f"Reinitialization attempt {attempt+1}/{CAMERA_MAX_RETRIES} failed. Retrying in {CAMERA_RETRY_DELAY} seconds.")
                        time.sleep(CAMERA_RETRY_DELAY)
                if not cap.isOpened():
                    logger.error("Failed to reinitialize camera. Exiting capture thread.")
                    stop_event.set()
                    return

            time.sleep(0.1)
            continue

        # Reset failure count on successful frame capture.
        consecutive_failures = 0
        frame_counter += 1
        if frame_counter % FRAME_SKIP != 0:
            continue

        # Save frame to a temporary file.
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir=UPLOADS_FOLDER)
            cv2.imwrite(temp_file.name, frame)
            temp_file.close()
        except Exception as e:
            logger.error(f"Error writing image: {e}")
            continue

        # If the queue is full, clear it so that only the latest frame remains.
        if image_queue.full():
            try:
                while not image_queue.empty():
                    old_frame = image_queue.get_nowait()
                    if os.path.exists(old_frame):
                        os.remove(old_frame)
                    logger.info("Cleared an old frame from queue.")
            except Exception as e:
                logger.error("Error clearing queue: " + str(e))

        try:
            image_queue.put(temp_file.name, timeout=1)
        except queue.Full:
            logger.warning("Queue still full, discarding frame.")
            try:
                os.remove(temp_file.name)
            except Exception as e:
                logger.error(f"Error deleting file {temp_file.name}: {e}")
        time.sleep(0.1)

    cap.release()
    logger.info("Capture thread exiting.")

def process_images():
    """
    Process images from the queue. Runs inference and updates the LCD if a detection occurs.
    Processes only the most recent frame in the queue.
    """
    global last_detection_time
    while not stop_event.is_set() or not image_queue.empty():
        if not button.check_button():
            time.sleep(0.1)
            continue

        try:
            # Drain the queue to keep only the most recent frame.
            latest_image = None
            while not image_queue.empty():
                latest_image = image_queue.get_nowait()
            if latest_image is None:
                continue
        except queue.Empty:
            continue

        current_time = time.time()
        if current_time - last_detection_time < COOLDOWN_PERIOD:
            try:
                if os.path.exists(latest_image):
                    os.remove(latest_image)
            except Exception as e:
                logger.error(f"Error deleting image {latest_image}: {e}")
            continue

        plant, disease = run_inference(latest_image)
        if disease not in ["Unknown", "Error"]:
            last_detection_time = current_time
            try:
                lcd.lcd_byte(lcd.LCD_LINE_1, lcd.LCD_CMD)
                lcd.lcd_string(plant, 2, 50)
                lcd.lcd_byte(lcd.LCD_LINE_2, lcd.LCD_CMD)
                lcd.lcd_string(disease, 2, 50)
                logger.info(f"Detected Plant: {plant} | Disease: {disease}")
            except Exception as e:
                logger.error(f"Error updating LCD: {e}")
            time.sleep(LCD_TIMEOUT)
            lcd.clear()

        try:
            if os.path.exists(latest_image):
                os.remove(latest_image)
        except Exception as e:
            logger.error(f"Error deleting image {latest_image}: {e}")

    logger.info("Processing thread exiting.")

def on_closing():
    """
    Cleanup function to stop threads, clear the LCD display, and remove temporary files.
    """
    stop_event.set()
    logger.info("Stopping threads, clearing LCD, and cleaning up uploads folder.")
    try:
        lcd.clear()
    except Exception as e:
        logger.error(f"Error clearing LCD: {e}")
    time.sleep(1)
    for filename in os.listdir(UPLOADS_FOLDER):
        file_path = os.path.join(UPLOADS_FOLDER, filename)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")

def main():
    logger.info("Waiting for button press to start live detection...")
    try:
        while True:
            if button.check_button():
                logger.info("Button pressed. Starting live detection!")
                stop_event.clear()
                capture_thread = threading.Thread(target=capture_frames)
                process_thread = threading.Thread(target=process_images)
                capture_thread.start()
                process_thread.start()

                while button.check_button():
                    if not process_thread.is_alive():
                        logger.error("Processing thread has exited unexpectedly. Stopping program.")
                        on_closing()
                        sys.exit(1)
                    time.sleep(0.1)

                logger.info("Button released. Stopping live detection...")
                on_closing()
                capture_thread.join()
                process_thread.join()
                logger.info("Threads successfully stopped.")
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt detected. Exiting...")
        on_closing()

if __name__ == "__main__":
    main()