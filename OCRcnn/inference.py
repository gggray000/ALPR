import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from pathlib import Path
import sys
from segmentation import preprocess

# Load model
model = load_model("/home/stud3/Desktop/ALPR/OCRcnn/ocr_plate_model.h5")
model.summary()
# Map label index to character
idx_to_char = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def preprocess_plate(plate_img):
    """ Convert license plate image to grayscale & binary """
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def segment_characters(thresh_img):
    """ Segment characters from license plate image """
    contours, _ = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_regions = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 0.4 * thresh_img.shape[0] and w > 5:
            char_regions.append((x, y, w, h))

    char_regions = sorted(char_regions, key=lambda b: b[0])  # left to right

    chars = []
    for x, y, w, h in char_regions:
        char_img = thresh_img[y:y+h, x:x+w]
        char_img = cv2.resize(char_img, (32, 32))
        char_img = char_img.astype("float32") / 255.0
        chars.append(char_img)

    return chars

def predict_characters(char_images):
    """ Predict each character using the trained model """
    result = ""
    for img in char_images:
        img = img.reshape(1, 32, 32, 1)
        pred = model.predict(img)
        char = idx_to_char[np.argmax(pred)]
        result += char
    return result

def recognize_plate(image_path):
    plate_img = cv2.imread(image_path)
    thresh = preprocess_plate(plate_img)
    char_images = segment_characters(thresh)
    plate_text = predict_characters(char_images)
    return plate_text

# ============================
# TEST ON ALL IMAGES IN plates/
plate_dir = r"/home/stud3/Desktop/ALPR/OCRcnn/input/"
for filename in os.listdir(plate_dir):

    if filename.lower().endswith((".png", ".jpg")):
        full_path = os.path.join(plate_dir, filename)
        text = recognize_plate(full_path)
        print(f"{filename}: {text}")
        
        
