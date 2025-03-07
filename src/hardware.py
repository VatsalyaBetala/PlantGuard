import RPi.GPIO as GPIO
import time

class PiController:
    def __init__(self, button_pin=18, red_pin=7, green_pin=5, blue_pin=3):
        """
        Initialize GPIO for both the button and the RGB LED.
        """
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        
        # Setup button
        self.button_pin = button_pin
        GPIO.setup(self.button_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        self.running = False
        self.button_pressed = False
        
        # Setup RGB LED pins and PWM
        self.red_pin = red_pin
        self.green_pin = green_pin
        self.blue_pin = blue_pin
        
        GPIO.setup(self.red_pin, GPIO.OUT)
        GPIO.setup(self.green_pin, GPIO.OUT)
        GPIO.setup(self.blue_pin, GPIO.OUT)
        
        self.red_pwm = GPIO.PWM(self.red_pin, 1000)   # 1 kHz frequency
        self.green_pwm = GPIO.PWM(self.green_pin, 1000)
        self.blue_pwm = GPIO.PWM(self.blue_pin, 1000)
        
        self.red_pwm.start(0)   # Start with LED off
        self.green_pwm.start(0)
        self.blue_pwm.start(0)
    
    def check_button(self):
        """
        Checks the button state and toggles the running flag when pressed.
        """
        state = GPIO.input(self.button_pin)
        if state == GPIO.HIGH and not self.button_pressed:
            self.button_pressed = True  # Prevent multiple toggles
            self.running = not self.running  # Toggle running state
            print("Program running..." if self.running else "Program stopped.")
        elif state == GPIO.LOW and self.button_pressed:
            self.button_pressed = False  # Reset after release
        return self.running
    
    def set_color(self, red_intensity, green_intensity, blue_intensity):
        """
        Sets the LED color with intensities (0-100%).
        """
        self.red_pwm.ChangeDutyCycle(red_intensity)
        self.green_pwm.ChangeDutyCycle(green_intensity)
        self.blue_pwm.ChangeDutyCycle(blue_intensity)
    
    def red(self, intensity=100):
        """Turn on red color with the given intensity (0-100%)."""
        self.set_color(intensity, 0, 0)
    
    def green(self, intensity=100):
        """Turn on green color with the given intensity (0-100%)."""
        self.set_color(0, intensity, 0)
    
    def blue(self, intensity=100):
        """Turn on blue color with the given intensity (0-100%)."""
        self.set_color(0, 0, intensity)
    
    def off(self):
        """Turn off the LED."""
        self.set_color(0, 0, 0)
    
    def cleanup(self):
        """
        Clean up the GPIO pins before exiting.
        """
        self.red_pwm.stop()
        self.green_pwm.stop()
        self.blue_pwm.stop()
        self.red_pwm = None
        self.green_pwm = None
        self.blue_pwm = None
        GPIO.cleanup()
