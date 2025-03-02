import os
import time
import uuid
import traceback
import cv2
import queue
import threading
import tempfile
import tkinter as tk
from PIL import Image, ImageTk
from gpiozero import Button, Buzzer
from dotenv import load_dotenv

# Import your inference functions and utilities
from src.utils import check_and_download_models
from src.inference import detect_leaf, classify_plant, classify_disease

# Load environment variables
load_dotenv()

# Ensure uploads folder exists (for temporary files)
UPLOADS_FOLDER = "uploads"
if not os.path.exists(UPLOADS_FOLDER):
    os.makedirs(UPLOADS_FOLDER)

# Check/download models if necessary
check_and_download_models()

# Hardware setup
button = Button(17)  # Button to start the system
buzzer = Buzzer(18)  # Buzzer for alerts

# Processing parameters
QUEUE_MAXSIZE = 10  # Maximum images waiting for processing
COOLDOWN_PERIOD = 10  # Cooldown period after a detection (seconds)
DISPLAY_TIME = 5  # How long to display the result window (seconds)

# Queues for inter-thread communication
image_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)  # For captured frames (file paths)
display_queue = queue.Queue()  # For display events (tuples of data)

# Event to signal threads to stop
stop_event = threading.Event()

# Global variable to track last detection time
last_detection_time = 0


def trigger_buzzer():
    """Activate the buzzer for a short period."""
    buzzer.on()
    time.sleep(0.5)
    buzzer.off()


def display_window(image_path, plant, disease):
    """
    Create a new Toplevel window (on the main thread) to display the image
    and, below it, display the plant and disease information.
    The window automatically closes after DISPLAY_TIME seconds.
    """
    # Create a new window using the main root as parent
    window = tk.Toplevel(root)
    window.title("Disease Detection Result")

    # Optionally, set a fixed window size (or let it auto-adjust)
    window.geometry("700x600")  # You can adjust these values as needed

    # Open the image using Pillow
    try:
        image = Image.open(image_path)
    except Exception as e:
        print("Error opening image:", e)
        return

    # Resize image to a fixed size (for example, 640x480) while preserving aspect ratio if desired
    target_width = 640
    target_height = 480
    image = image.resize((target_width, target_height))

    # Convert the modified image to a Tkinter PhotoImage
    photo = ImageTk.PhotoImage(image)
    img_label = tk.Label(window, image=photo)
    img_label.image = photo  # keep a reference to prevent garbage collection
    img_label.pack(padx=10, pady=10)

    # Create a label below the image to display the plant and disease information
    result_text = f"Plant: {plant} | Disease: {disease}"
    text_label = tk.Label(window, text=result_text, font=("Helvetica", 16))
    text_label.pack(pady=10)

    # Automatically close the window after DISPLAY_TIME seconds and then delete the file
    def close_window():
        window.destroy()
        try:
            os.remove(image_path)
        except Exception as e:
            print(f"Error deleting image file {image_path}: {e}")

    window.after(DISPLAY_TIME * 1000, close_window)

def poll_display_queue():
    """
    Poll the display queue periodically and display results (in the main thread).
    """
    try:
        while not display_queue.empty():
            image_path, plant, disease = display_queue.get_nowait()
            display_window(image_path, plant, disease)
    except queue.Empty:
        pass
    root.after(100, poll_display_queue)


def run_inference(image_path):
    """
    Run the inference pipeline: detect and crop the leaf, classify the plant,
    and determine the disease.
    """
    cropped_leaf_path = detect_leaf(image_path)
    if cropped_leaf_path is None:
        return "No Leaf Detected", "Unknown"
    plant = classify_plant(cropped_leaf_path)
    disease = classify_disease(cropped_leaf_path, plant)
    return plant, disease


def capture_frames():
    cap = cv2.VideoCapture(0)  # Use default camera (adjust if necessary)
    frame_skip = 20  # Process every 20th frame
    frame_counter = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            frame_counter += 1
            # Only process every frame_skip-th frame
            if frame_counter % frame_skip != 0:
                continue

            # Create a temporary file in the uploads folder
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
    Process images from the queue. If a detection is made (i.e. disease is not "Unknown")
    and the cooldown period has passed, trigger the buzzer and send the display request
    to the main thread via the display queue.
    """
    global last_detection_time
    while not stop_event.is_set() or not image_queue.empty():
        try:
            image_path = image_queue.get(timeout=1)
        except queue.Empty:
            continue

        current_time = time.time()
        # Enforce cooldown period to prevent redundant processing
        if current_time - last_detection_time < COOLDOWN_PERIOD:
            try:
                os.remove(image_path)
            except Exception as e:
                print(f"Error deleting temp file: {e}")
            continue

        try:
            plant, disease = run_inference(image_path)
            if disease != "Unknown":
                last_detection_time = current_time
                trigger_buzzer()
                # Send display request to the main thread via display_queue
                display_queue.put((image_path, plant, disease))
            else:
                os.remove(image_path)
        except Exception as e:
            print("Error during inference processing:")
            print(traceback.format_exc())
            try:
                os.remove(image_path)
            except Exception as e:
                print(f"Error deleting temp file: {e}")


def on_closing():
    """Cleanup function called when the main window is closed."""
    stop_event.set()
    # Clean up all files in the uploads folder
    for filename in os.listdir(UPLOADS_FOLDER):
        file_path = os.path.join(UPLOADS_FOLDER, filename)
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
    root.destroy()

if __name__ == "__main__":
    print("Waiting for button press to start live detection...")
    button.wait_for_press()
    print("Button pressed. Starting live detection!")

    # Create the main Tkinter root window on the main thread.
    root = tk.Tk()
    root.withdraw()  # Hide the root window; we'll create Toplevel windows for display.
    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Start the capture and processing threads
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    process_thread = threading.Thread(target=process_images, daemon=True)
    capture_thread.start()
    process_thread.start()

    # Start polling the display queue from the main thread.
    root.after(100, poll_display_queue)
    root.mainloop()