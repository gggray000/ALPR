import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io

def create_app(test_config=None):
    app = Flask(__name__)
    CORS(app)

    model = tf.keras.models.load_model("./handwritten.keras")

    @app.route('/predict', methods=['POST'])
    def predictDigit():
        file=request.files.get('image')
        if not file:
            return jsonify({'error':'No image uploaded'}), 400
        
        image = Image.open(io.BytesIO(file.read()))
        prediction = predict(model, image)
        return jsonify({'prediction':int(prediction)})

    return app

def preprocess_image(img):
    # Load the image in grayscale
    img = img.convert("L")
    img = np.array(img)

    # Resize if necessary (should be 28x28 for MNIST)
    if img.shape != (28, 28):
        img = cv2.resize(img, (28, 28))

    # Normalize and reshape to match model input
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # Shape: (1, 28, 28)
    
    return img

def predict(model, image):
    
    img = preprocess_image(image)
    prediction = model.predict(img)
    predicted_digit = np.argmax(prediction)

    return predicted_digit

if __name__ == "__main__":
    app=create_app()
    app.run(host="127.0.0.1", port=5002)