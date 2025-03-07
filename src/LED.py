import RPi.GPIO as GPIO
import time

class RGBLed:
    def __init__(self, red_pin=7, green_pin=5, blue_pin=3):
        """
        Initialize the RGB LED with PWM for intensity control.
        """
        self.red_pin = red_pin
        self.green_pin = green_pin
        self.blue_pin = blue_pin
        
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        
        GPIO.setup(self.red_pin, GPIO.OUT)
        GPIO.setup(self.green_pin, GPIO.OUT)
        GPIO.setup(self.blue_pin, GPIO.OUT)
        
        self.red_pwm = GPIO.PWM(self.red_pin, 1000)  # 1 kHz frequency
        self.green_pwm = GPIO.PWM(self.green_pin, 1000)
        self.blue_pwm = GPIO.PWM(self.blue_pin, 1000)
        
        self.red_pwm.start(0)  # Start with 0% duty cycle (off)
        self.green_pwm.start(0)
        self.blue_pwm.start(0)
    
    def set_color(self, red_intensity, green_intensity, blue_intensity):
        """
        Set the RGB LED color with intensity (0-100%).
        """
        self.red_pwm.ChangeDutyCycle(red_intensity)
        self.green_pwm.ChangeDutyCycle(green_intensity)
        self.blue_pwm.ChangeDutyCycle(blue_intensity)
        print(f"LED Color set to: Red={red_intensity}%, Green={green_intensity}%, Blue={blue_intensity}%")

    def red(self, intensity=100):
        """Turn on red color with given intensity (0-100%)."""
        self.set_color(intensity, 0, 0)

    def green(self, intensity=100):
        """Turn on green color with given intensity (0-100%)."""
        self.set_color(0, intensity, 0)

    def blue(self, intensity=100):
        """Turn on blue color with given intensity (0-100%)."""
        self.set_color(0, 0, intensity)
    
    def off(self):
        """Turn off the LED."""
        self.set_color(0, 0, 0)

    def cleanup(self):
        """
        Clean up the GPIO pins before exiting.
        """
        # Stop all PWM channels
        self.red_pwm.stop()
        self.green_pwm.stop()
        self.blue_pwm.stop()
        
        # Remove references to PWM objects to prevent further __del__ calls
        self.red_pwm = None
        self.green_pwm = None
        self.blue_pwm = None
        
        # Finally, clean up the GPIO settings
        GPIO.cleanup()

