from flask import Flask, render_template, request
from PIL import Image
import pytesseract

app = Flask(__name__)

def perform_ocr(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        file = request.files['file']
        if file:
            # Save the uploaded image
            image_path = '/home/sirumavills/Wonders/ocr/simple_ocr/test.jpg'
            file.save(image_path)

            # Perform OCR on the image
            result = perform_ocr(image_path)

            return render_template('index.html', result=result, image_path=image_path)

    return render_template('index.html', result=None, image_path=None)

if __name__ == "__main__":
    app.run(debug=True)
