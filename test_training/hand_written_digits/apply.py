import numpy as np
import cv2
import tensorflow as tf

def preprocess_image(img_path):
    # Load the image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Image at {img_path} not found.")

    # Resize if necessary (should be 28x28 for MNIST)
    if img.shape != (28, 28):
        img = cv2.resize(img, (28, 28))

    # Invert colors (if background is white and digit is black)
    img = np.invert(img)

    # Normalize and reshape to match model input
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # Shape: (1, 28, 28)
    
    return img

def predict(img_path, model_path):
    model = tf.keras.models.load_model(model_path)
    img = preprocess_image(img_path)
    prediction = model.predict(img)
    predicted_digit = np.argmax(prediction)

    print(f"Predicted digit: {predicted_digit}")

if __name__ == "__main__":
    predict(f"/home/stud3/Desktop/ALPR/test_training/hand_written_digits/digits/digit2.png", 
            f"/home/stud3/Desktop/ALPR/test_training/hand_written_digits/handwritten.keras")
    
# For flask, we need a preprocessing method, to read image from file stream using PIL Image, then do resizing.
# Then return the inference result with json format.
# One endpoint should be enough.