from PIL import Image
import pytesseract

def ocr(image_path):
    # Open the image file
    img = Image.open(image_path)

    # Perform OCR using pytesseract
    text = pytesseract.image_to_string(img)

    return text

if __name__ == "__main__":
    # Replace 'your_image.png' with the path to your image file
    image_path = 'test.jpg'
    
    result = ocr(image_path)
    
    print(f"OCR Result: {result}")
