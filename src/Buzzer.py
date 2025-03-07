import RPi.GPIO as GPIO
import time

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)  # Set to BOARD or BCM as needed

class Buzzer:
    def __init__(self, buzzer_pin=2):
        self.buzzer_pin = buzzer_pin
        GPIO.setup(self.buzzer_pin, GPIO.OUT)
        GPIO.output(self.buzzer_pin, GPIO.LOW)
    
    def buzz(self, duration=0.5):
        GPIO.output(self.buzzer_pin, GPIO.HIGH)
        print("Buzzer ON")
        time.sleep(duration)
        GPIO.output(self.buzzer_pin, GPIO.LOW)