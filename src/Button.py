import RPi.GPIO as GPIO
import time

class Button:
    def __init__(self, button_pin=18):
        """
        Initialize the button.
        """
        self.button_pin = button_pin
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.button_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

        self.running = False  # Track the running state
        self.button_pressed = False

    def check_button(self):
        state = GPIO.input(self.button_pin)
        if state == GPIO.HIGH and not self.button_pressed:
            self.button_pressed = True  # Avoid multiple detections
            self.running = not self.running  # Toggle state
            print("Program running..." if self.running else "Program stopped.")
        elif state == GPIO.LOW and self.button_pressed:
            self.button_pressed = False  # Reset button state
        return self.running
