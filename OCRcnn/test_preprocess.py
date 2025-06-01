from tensorflow.keras.models import load_model
from segmentation import preprocess
import numpy as np

model = load_model("/home/stud3/Desktop/ALPR/OCRcnn/ocr_plate_model.h5")
idx_to_char = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

model.summary()
path = "/home/stud3/Desktop/ALPR/OCRcnn/input/plate7.jpg"
filename = "plate7.jpg"
print(path)

result = ""
characters = preprocess(path)
for c in characters:
     pred = model.predict(np.expand_dims(c, axis=0)) # shape of c: (1, 32, 32, 1)
     char = idx_to_char[np.argmax(pred)]
     result += char

print(f"{filename}: {result}")