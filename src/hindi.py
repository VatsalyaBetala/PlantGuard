import time
from PIL import Image, ImageDraw, ImageFont
import Adafruit_SSD1306

# OLED display dimensions (change these if your display is different)
OLED_WIDTH = 128
OLED_HEIGHT = 64

def create_hindi_image():
    """
    Creates an image with Hindi text rendered in the center.
    """
    # Create a new image with a white background
    image = Image.new('RGB', (OLED_WIDTH, OLED_HEIGHT), 'white')
    draw = ImageDraw.Draw(image)
    
    # Hindi text to display ("How are you?" in Hindi)
    hindi_text = "आप कैसे हैं?"
    
    # Specify a font that supports Hindi (update the path if necessary)
    font_path = "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"
    font_size = 16
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print("Hindi font not found. Falling back to default font (which may not support Hindi).")
        font = ImageFont.load_default()
    
    # Calculate position to center the text on the image
    text_width, text_height = draw.textsize(hindi_text, font=font)
    x = (OLED_WIDTH - text_width) // 2
    y = (OLED_HEIGHT - text_height) // 2
    
    # Draw the Hindi text onto the image in black color
    draw.text((x, y), hindi_text, font=font, fill='black')
    
    return image

def display_image(image):
    """
    Initializes the SSD1306 OLED display, converts the image to 1-bit mode,
    and displays it.
    """
    # Initialize the display (adjust rst and other parameters as needed for your hardware)
    disp = Adafruit_SSD1306.SSD1306_128_64(rst=None)
    disp.begin()
    disp.clear()
    disp.display()
    
    # Convert image to 1-bit color required by many OLED displays
    image_monochrome = image.convert('1')
    
    # Display the image
    disp.image(image_monochrome)
    disp.display()

def clear_display():
    """
    Clears the OLED display.
    """
    disp = Adafruit_SSD1306.SSD1306_128_64(rst=None)
    disp.begin()
    disp.clear()
    disp.display()

def main():
    # Create an image with Hindi text
    hindi_image = create_hindi_image()
    
    # (Optional) Open the image using your system's default image viewer for testing
    hindi_image.show()
    
    # Small delay before sending to the OLED
    time.sleep(2)
    
    # Display the Hindi image on the OLED
    display_image(hindi_image)
    
    # Keep the image on the display for 5 seconds
    time.sleep(5)
    
    # Clear the display after the delay
    clear_display()

if __name__ == '__main__':
    main()