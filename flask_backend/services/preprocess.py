import numpy as np
import cv2

def preprocess_image(pil_image):
    # Convert to grayscale
    img = pil_image.convert("L")
    img = np.array(img)

    # Resize to 28x28
    if img.shape != (28, 28):
        img = cv2.resize(img, (28, 28))

    # Invert colors if needed (white background)
    img = np.invert(img)

    # Normalize and reshape
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # Shape: (1, 28, 28)
    return img