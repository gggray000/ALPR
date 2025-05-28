import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image

# Create vocabulary from filenames
def build_vocab(filenames):
    all_chars = set("".join(filenames))
    return {c: i for i, c in enumerate(sorted(all_chars))}

# Load dataset
def load_dataset_from_filenames(folder_path, image_size=(256, 32), char2idx=None):
    images = []
    labels = []
    filenames = []

    for file in os.listdir(folder_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(folder_path, file)
            label = os.path.splitext(file)[0]
            img = Image.open(filepath).convert('L').resize(image_size)
            img = np.array(img) / 255.0
            images.append(img)
            labels.append(label)
            filenames.append(label)

    images = np.expand_dims(np.array(images, dtype=np.float32), axis=-1)

    if char2idx is None:
        char2idx = build_vocab(filenames)

    encoded_labels = [[char2idx[c] for c in label] for label in labels]
    label_lengths = [len(label) for label in encoded_labels]
    max_len = max(label_lengths)

    padded_labels = tf.keras.preprocessing.sequence.pad_sequences(
        encoded_labels, maxlen=max_len, padding='post'
    )

    return images, padded_labels, label_lengths, char2idx

# CTC loss function
def ctc_loss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.math.count_nonzero(y_true, axis=1, dtype="int64")
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = tf.expand_dims(label_length, axis=1)
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

# Build CNN + BiLSTM + CTC model
def build_model(input_shape, vocab_size):
    inputs = layers.Input(shape=input_shape)

    # Use strided convs to downsample only in height
    x = layers.Conv2D(32, (3, 3), strides=(2, 1), padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, (3, 3), strides=(2, 1), padding='same', activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)

    # Reshape to (batch, time_steps, features)
    b, h, w, c = x.shape  # symbolic dimensions
    x = layers.Reshape((w, h * c))(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dense(vocab_size + 1, activation='softmax')(x)  # +1 for CTC blank

    return models.Model(inputs, x)

def train():
    train_path = '/home/stud3/Desktop/test_model_training/test_training/basic_CNN/dataset/train'
    val_path = '/home/stud3/Desktop/test_model_training/test_training/basic_CNN/dataset/val'

    train_images, train_labels, _, char2idx = load_dataset_from_filenames(train_path)
    val_images, val_labels, _, _ = load_dataset_from_filenames(val_path, char2idx=char2idx)

    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(32)
    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(32)

    input_shape = (32, 256, 1)
    vocab_size = len(char2idx)

    model = build_model(input_shape, vocab_size)
    model.compile(optimizer='adam', loss=ctc_loss)
    model.summary()

    model.fit(train_ds, validation_data=val_ds, epochs=300)
    model.save('/home/stud3/Desktop/test_model_training/test_training/basic_CNN/plate_CNN_ocr.keras')

def apply(model_path, img_path):
    # Hard-coded vocab — make sure it's the same as training vocab
    char2idx = build_vocab("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    idx2char = {v: k for k, v in char2idx.items()}

    model = tf.keras.models.load_model(model_path, compile=False)

    img = cv2.imread(f"{img_path}", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 32))
    img = img.astype(np.float32) / 255.0
    img = img.reshape(1, 32, 256, 1)

    prediction = model.predict(img)
    input_len = np.ones((1,)) * prediction.shape[1]
    decoded = tf.keras.backend.ctc_decode(prediction, input_length=input_len, greedy=True)[0][0]
    result = ''.join([idx2char.get(i, '') for i in decoded.numpy()[0] if i > 0])

    print(f"Predicted: {result}")
    plt.imshow(img[0, :, :, 0], cmap='gray')
    plt.title(f"Predicted: {result}")
    plt.show()

if __name__ == '__main__':
    #train()
    apply(
    '/home/stud3/Desktop/test_model_training/test_training/basic_CNN/plate_CNN_ocr.keras', 
    '/home/stud3/Desktop/test_model_training/test_training/basic_CNN/input/2.jpg'
    )