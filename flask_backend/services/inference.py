import tensorflow as tf
import numpy as np
from services.preprocess import preprocess_image

MODEL_PATH = 'models/handwritten.keras'
model = tf.keras.models.load_model(MODEL_PATH)

def predict_digit(image):
    processed = preprocess_image(image)
    prediction = model.predict(processed)
    return np.argmax(prediction)