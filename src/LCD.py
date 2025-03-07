import RPi.GPIO as GPIO
import time

class LCD:
    def __init__(self):
        # Define GPIO to LCD mapping (BOARD mode pin numbers)
        if not GPIO.getmode():
            GPIO.setmode(GPIO.BOARD)
        self.LCD_RS = 37  # GPIO 26 -> BOARD 37
        self.LCD_E  = 35  # GPIO 19 -> BOARD 35
        self.LCD_D4 = 33  # GPIO 13 -> BOARD 33
        self.LCD_D5 = 31  # GPIO 6  -> BOARD 31
        self.LCD_D6 = 29  # GPIO 5  -> BOARD 29
        self.LCD_D7 = 23  # GPIO 11 -> BOARD 23
        self.LED_ON = 10  # GPIO 15 -> BOARD 10
        self.PWM_PIN = 32  # GPIO 12 -> BOARD 32
        
        # Define some device constants
        self.LCD_WIDTH = 16    # Maximum characters per line
        self.LCD_CHR = True
        self.LCD_CMD = False

        self.LCD_LINE_1 = 0x80 # LCD RAM address for the 1st line
        self.LCD_LINE_2 = 0xC0 # LCD RAM address for the 2nd line 

        # Timing constants
        self.E_PULSE = 0.00005
        self.E_DELAY = 0.00005
        self.lcd_init()
    def lcd_init(self): # Use BOARD pin numbers
        GPIO.setup(self.LCD_E, GPIO.OUT)  # E
        GPIO.setup(self.LCD_RS, GPIO.OUT) # RS
        GPIO.setup(self.LCD_D4, GPIO.OUT) # DB4
        GPIO.setup(self.LCD_D5, GPIO.OUT) # DB5
        GPIO.setup(self.LCD_D6, GPIO.OUT) # DB6
        GPIO.setup(self.LCD_D7, GPIO.OUT) # DB7
        GPIO.setup(self.LED_ON, GPIO.OUT) # Backlight enable  
        GPIO.setup(self.PWM_PIN, GPIO.OUT) # PWM for contrast
        # Initialise display
        self.lcd_byte(0x33, self.LCD_CMD)
        time.sleep(0.005)
        self.lcd_byte(0x32, self.LCD_CMD)
        time.sleep(0.005)
        self.lcd_byte(0x28, self.LCD_CMD)
        time.sleep(0.005)
        self.lcd_byte(0x0C, self.LCD_CMD)  
        time.sleep(0.005)
        self.lcd_byte(0x06, self.LCD_CMD)
        time.sleep(0.005)
        self.lcd_byte(0x01, self.LCD_CMD)
        time.sleep(0.005)

        # Initialize PWM
        # Initialize PWM only if it hasn't been initialized before
        if not hasattr(self, 'pwm') or self.pwm is None:
            self.pwm = GPIO.PWM(self.PWM_PIN, 1000)
            self.pwm.start(0)
    def set_brightness(self, percentage):
        """Set the brightness of the LCD backlight as a percentage from 0 to 100."""
        duty_cycle = max(0, min(100, percentage))
        self.pwm.ChangeDutyCycle(duty_cycle)
    
    def lcd_string(self, message, style, brightness):
        """Send string to display with specified brightness.
        style=1 Left justified
        style=2 Centred
        style=3 Right justified
        brightness=brightness percentage (0-100)
        """
        # Set brightness
        self.set_brightness(brightness)

        if style == 1:
            message = message.ljust(self.LCD_WIDTH, " ")  
        elif style == 2:
            message = message.center(self.LCD_WIDTH, " ")
        elif style == 3:
            message = message.rjust(self.LCD_WIDTH, " ")

        for i in range(self.LCD_WIDTH):
            self.lcd_byte(ord(message[i]), self.LCD_CHR)
    
    def lcd_byte(self, bits, mode):
        """Send byte to data pins
        bits = data
        mode = True for character
               False for command
        """
        GPIO.output(self.LCD_RS, mode)  # RS

        # High bits
        GPIO.output(self.LCD_D4, False)
        GPIO.output(self.LCD_D5, False)
        GPIO.output(self.LCD_D6, False)
        GPIO.output(self.LCD_D7, False)
        if bits & 0x10 == 0x10:
            GPIO.output(self.LCD_D4, True)
        if bits & 0x20 == 0x20:
            GPIO.output(self.LCD_D5, True)
        if bits & 0x40 == 0x40:
            GPIO.output(self.LCD_D6, True)
        if bits & 0x80 == 0x80:
            GPIO.output(self.LCD_D7, True)

        # Toggle 'Enable' pin
        time.sleep(self.E_DELAY)    
        GPIO.output(self.LCD_E, True)  
        time.sleep(self.E_PULSE)
        GPIO.output(self.LCD_E, False)  
        time.sleep(self.E_DELAY)      

        # Low bits
        GPIO.output(self.LCD_D4, False)
        GPIO.output(self.LCD_D5, False)
        GPIO.output(self.LCD_D6, False)
        GPIO.output(self.LCD_D7, False)
        if bits & 0x01 == 0x01:
            GPIO.output(self.LCD_D4, True)
        if bits & 0x02 == 0x02:
            GPIO.output(self.LCD_D5, True)
        if bits & 0x04 == 0x04:
            GPIO.output(self.LCD_D6, True)
        if bits & 0x08 == 0x08:
            GPIO.output(self.LCD_D7, True)

        # Toggle 'Enable' pin
        time.sleep(self.E_DELAY)    
        GPIO.output(self.LCD_E, True)  
        time.sleep(self.E_PULSE)
        GPIO.output(self.LCD_E, False)  
        time.sleep(self.E_DELAY)   
    def clear(self):
        self.lcd_byte(0x01, self.LCD_CMD)
        time.sleep(0.005)  # Small delay to allow the command to process