# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
#
# # Dataset path
# DATASET_DIR = r"/dataset"
#
# # Parameters
# IMG_SIZE = (32, 32)
# BATCH_SIZE = 64
# EPOCHS = 25
#
# # Training data with light augmentation
# datagen = ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.2,
#     rotation_range=10,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.1,
#     zoom_range=0.1,
# )
#
# train_generator = datagen.flow_from_directory(
#     DATASET_DIR,
#     target_size=IMG_SIZE,
#     color_mode="grayscale",
#     batch_size=BATCH_SIZE,
#     class_mode="categorical",
#     subset="training",
#     shuffle=True,
# )
#
# validation_generator = datagen.flow_from_directory(
#     DATASET_DIR,
#     target_size=IMG_SIZE,
#     color_mode="grayscale",
#     batch_size=BATCH_SIZE,
#     class_mode="categorical",
#     subset="validation",
#     shuffle=False,
# )
#
# # Number of classes
# num_classes = len(train_generator.class_indices)
# print(f"Number of classes: {num_classes}")
#
# # Build CNN model
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
#     MaxPooling2D(2, 2),
#
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#
#     Dense(num_classes, activation='softmax')
# ])
#
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()
#
# # Training
# early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
#
# history = model.fit(
#     train_generator,
#     epochs=EPOCHS,
#     validation_data=validation_generator,
#     callbacks=[early_stop]
# )
#
# # Save the model
# model.save("ocr_cnn_model.h5")
# print("Model saved: ocr_cnn_model.h5")

import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

# List of license plate characters: 0-9 + A-Z
classes = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
char_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

# Read data
def load_data(dataset_path, img_size=32):
    X, y = [], []
    for label in os.listdir(dataset_path):
        if label not in char_to_idx:
            continue
        label_path = os.path.join(dataset_path, label)
        for file in os.listdir(label_path):
            img = cv2.imread(os.path.join(label_path, file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size))
            img = img / 255.0  # normalize
            X.append(img)
            y.append(char_to_idx[label])
    return np.array(X), np.array(y)

# Load data
X, y = load_data(r"/home/stud3/Desktop/test_model_training/pretrained/dataset")  # ← Change this to your dataset path
X = X.reshape(-1, 32, 32, 1)
y = to_categorical(y, num_classes=len(classes))

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(classes), activation='softmax')
])

# Compile & train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test))

# Save model
model.save("ocr_plate_model.h5")

