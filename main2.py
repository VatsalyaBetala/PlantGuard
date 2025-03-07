import RPi.GPIO as GPIO
import time
import threading
import queue
import cv2
import tempfile
import logging
import sys
import os
from dotenv import load_dotenv
from src.LCD import LCD
from src.utils import check_and_download_models
from src.inference import detect_leaf, classify_plant, classify_disease
from src.hardware import PiController

# -----------------------------------------------------------------------------
# Load Environment Variables and Logging Setup
# -----------------------------------------------------------------------------
load_dotenv()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

file_handler = logging.FileHandler('app.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Configuration parameters (with defaults)
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

if not os.path.exists(UPLOADS_FOLDER):
    os.makedirs(UPLOADS_FOLDER)

# Uncomment if needed: check_and_download_models()

# Create a queue for inter-thread communication
image_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)

last_detection_time = 0

# Initialize hardware interfaces:
# - Use PiController for both button and LED control.
controller = PiController(button_pin=BUTTON_PIN)
lcd = LCD()
lcd.clear()

# Event to signal threads to stop
stop_event = threading.Event()


# -----------------------------------------------------------------------------
# Inference & Image Processing Functions
# -----------------------------------------------------------------------------
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
    If there are camera errors (failure to capture frames or reinitialization issues),
    the LED is set to blue.
    """
    consecutive_failures = 0
    cap = None

    # Attempt initial camera connection with retries.
    for attempt in range(CAMERA_MAX_RETRIES):
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            logger.info("Camera connection established.")
            break
        else:
            logger.error(f"Attempt {attempt+1}/{CAMERA_MAX_RETRIES}: Could not open camera. Retrying in {CAMERA_RETRY_DELAY} seconds.")
            time.sleep(CAMERA_RETRY_DELAY)

    if cap is None or not cap.isOpened():
        logger.error("Max retries reached. Could not open the camera. Exiting capture thread.")
        controller.blue(100)
        stop_event.set()
        return

    frame_counter = 0
    while not stop_event.is_set():
        # Process frames only when running
        if not controller.running:
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            # Set LED to blue when a frame capture fails.
            controller.blue(100)
            logger.warning("Frame capture failed. Attempting to reinitialize camera.")
            consecutive_failures += 1
            if consecutive_failures >= CAMERA_MAX_CONSECUTIVE_FAILURES:
                logger.error("Too many consecutive frame failures. Reinitializing camera connection.")
                cap.release()
                # Indicate error state while reinitializing.
                controller.blue(100)
                for attempt in range(CAMERA_MAX_RETRIES):
                    cap = cv2.VideoCapture(0)
                    if cap.isOpened():
                        logger.info("Camera reinitialized successfully.")
                        consecutive_failures = 0
                        # Restore running colour (green) if detection is active.
                        if controller.running:
                            controller.green(100)
                        break
                    else:
                        logger.error(f"Reinitialization attempt {attempt+1}/{CAMERA_MAX_RETRIES} failed. Retrying in {CAMERA_RETRY_DELAY} seconds.")
                        controller.blue(100)
                        time.sleep(CAMERA_RETRY_DELAY)
                if not cap.isOpened():
                    logger.error("Failed to reinitialize camera. Exiting capture thread.")
                    stop_event.set()
                    return
            time.sleep(0.1)
            continue

        # On successful frame capture, if detection is active, set LED to green.
        if controller.running:
            controller.green(100)
        consecutive_failures = 0
        frame_counter += 1
        if frame_counter % FRAME_SKIP != 0:
            continue

        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir=UPLOADS_FOLDER)
            cv2.imwrite(temp_file.name, frame)
            temp_file.close()
        except Exception as e:
            logger.error(f"Error writing image: {e}")
            continue

        # Clear old frames if the queue is full.
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
    """
    global last_detection_time
    while not stop_event.is_set() or not image_queue.empty():
        if not controller.running:
            time.sleep(0.1)
            continue

        try:
            latest_image = None
            # Drain the queue and keep only the most recent image.
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


# -----------------------------------------------------------------------------
# Cleanup Function
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Main Function: Integrates Button/LED control with Inference Threads.
# -----------------------------------------------------------------------------
def main():
    # Set default state to red (indicating the program is stopped).
    controller.red(100)
    logger.info("Waiting for button press to start live detection...")
    try:
        while True:
            # Continuously poll the button to update the running state.
            controller.check_button()
            if controller.running:
                logger.info("Button pressed. Starting live detection!")
                # Set LED to green while running.
                controller.green(100)
                stop_event.clear()
                capture_thread = threading.Thread(target=capture_frames)
                process_thread = threading.Thread(target=process_images)
                capture_thread.start()
                process_thread.start()

                # While running, continue polling for button state changes.
                while controller.running:
                    controller.check_button()  # Allow toggling off.
                    if not process_thread.is_alive():
                        logger.error("Processing thread has exited unexpectedly. Stopping program.")
                        controller.blue(100)  # Error state indicated by blue.
                        on_closing()
                        sys.exit(1)
                    time.sleep(0.1)

                logger.info("Button pressed again. Stopping live detection...")
                # When detection stops (button pressed to stop), show red.
                controller.red(100)
                on_closing()
                capture_thread.join()
                process_thread.join()
                logger.info("Threads successfully stopped.")
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt detected. Exiting...")
        controller.blue(100)  # Error state indicated by blue.
        time.sleep(2)        # Allow time to see the blue LED
        on_closing()
    finally:
        # Turn LED off when fully exiting.
        controller.off()
        controller.cleanup()

if __name__ == "__main__":
    main()