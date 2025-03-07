import os
import time
import threading
import queue
import cv2
import tempfile
from dotenv import load_dotenv
from src.LCD import LCD
from src.Button import Button  # Import the button class
from src.utils import check_and_download_models
from src.inference import detect_leaf, classify_plant, classify_disease

# Load environment variables
load_dotenv()

# Ensure uploads folder exists
UPLOADS_FOLDER = "uploads"
if not os.path.exists(UPLOADS_FOLDER):
    os.makedirs(UPLOADS_FOLDER)

# Check/download models if necessary
#check_and_download_models()

# Processing parameters
QUEUE_MAXSIZE = 10  # Maximum images waiting for processing
COOLDOWN_PERIOD = 10  # Cooldown period after a detection (seconds)

# Queues for inter-thread communication
image_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)

# Global variable to track last detection time
last_detection_time = 0

# Initialize the button
button = Button(button_pin=18)

# Event to signal threads to stop
stop_event = threading.Event()
lcd = LCD()
lcd.clear()

def run_inference(image_path):
    """
    Runs the inference pipeline: detect leaf, classify plant, classify disease.
    """
    cropped_leaf_path = detect_leaf(image_path)
    if cropped_leaf_path is None:
        return "No Leaf Detected", "Unknown"
    
    plant = classify_plant(cropped_leaf_path)
    disease = classify_disease(cropped_leaf_path, plant)
    
    return plant, disease


def capture_frames():
    """
    Capture frames from the camera and add them to the queue.
    """
    phone_cam_url = "http://10.10.105.29:4747/video"
    cap = cv2.VideoCapture(phone_cam_url)  # Use default camera

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        stop_event.set()  # Signal to stop the program
        return

    frame_skip = 20  # Process every 20th frame
    frame_counter = 0

    while not stop_event.is_set():
        if not button.check_button():  # Stop capturing if button is toggled off
            time.sleep(0.1)
            continue
        
        ret, frame = cap.read()
        if not ret:
            print("Warning: Frame capture failed. Retrying...")
            time.sleep(0.1)
            continue

        frame_counter += 1
        if frame_counter % frame_skip != 0:
            continue

        # Save captured image
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir=UPLOADS_FOLDER)
        cv2.imwrite(temp_file.name, frame)
        temp_file.close()
        
        try:
            image_queue.put(temp_file.name, timeout=1)
        except queue.Full:
            os.remove(temp_file.name)

        time.sleep(0.1)

    cap.release()

def process_images():
    """
    Process images from the queue. If detection is made, print result.
    """
    global last_detection_time
    while not stop_event.is_set() or not image_queue.empty():
        if not button.check_button():  # Stop processing if button is toggled off
            time.sleep(0.1)
            continue
        
        try:
            image_path = image_queue.get(timeout=1)
        except queue.Empty:
            continue

        current_time = time.time()
        if current_time - last_detection_time < COOLDOWN_PERIOD:
            os.remove(image_path)
            continue

        try:
            plant, disease = run_inference(image_path)
            if disease != "Unknown":
                last_detection_time = current_time
                lcd.lcd_byte(lcd.LCD_LINE_1, lcd.LCD_CMD)
                lcd.lcd_string(plant, 2, 50)
                lcd.lcd_byte(lcd.LCD_LINE_2, lcd.LCD_CMD)
                lcd.lcd_string(disease, 2, 50)
                print(f"\nDetected Plant: {plant} | Disease: {disease}\n")
                time.sleep(5)
                lcd.clear()
            
            os.remove(image_path)

        except Exception as e:
            print("Error during inference processing:", e)
            os.remove(image_path)


def on_closing():
    """
    Cleanup function when stopping the script.
    """
    stop_event.set()
    for filename in os.listdir(UPLOADS_FOLDER):
        file_path = os.path.join(UPLOADS_FOLDER, filename)
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")


if __name__ == "__main__":
    print("Waiting for button press to start live detection...")

    try:
        while True:
            if button.check_button():  # Wait for button press
                print("Button pressed. Starting live detection!")

                # Start the capture and processing threads
                stop_event.clear()
                capture_thread = threading.Thread(target=capture_frames, daemon=True)
                process_thread = threading.Thread(target=process_images, daemon=True)
                capture_thread.start()
                process_thread.start()

                while button.check_button():  # Keep running while the button is on
                    time.sleep(0.1)

                print("\nStopping live detection...")
                on_closing()

            time.sleep(0.1)  # Prevent excessive CPU usage

    except KeyboardInterrupt:
        print("\nExiting...")
        on_closing()
