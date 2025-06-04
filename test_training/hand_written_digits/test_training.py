import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import nvtx
import mlflow

mlflow.set_tracking_uri(uri="http://localhost:5000")
mlflow.set_experiment("hand_written_digits")

class MLflowLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for key, value in logs.item():
                mlfow.log_metric(key, value, step=epoch)

# 1. Create vocabulary
def build_vocab(filenames):
    all_chars = set("".join(filenames))
    return {c: i + 1 for i, c in enumerate(sorted(all_chars))}  # +1 to reserve 0 for padding

# 2. Load images and labels
@nvtx.annotate("Loading Training Data", color="blue")
def load_dataset_from_filenames(folder_path, image_size=(128, 32)):
    images = []
    labels = []
    filenames = []
    
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(folder_path, file)
            label = os.path.splitext(file)[0]  # Get filename without extension

            img = Image.open(filepath).convert('L').resize(image_size)  # Grayscale + resize
            img = np.array(img) / 255.0  # Normalize
            images.append(img)
            labels.append(label)
            filenames.append(label)

    # Convert to arrays
    images = np.expand_dims(np.array(images, dtype=np.float32), axis=-1)  # (N, H, W, 1)

    # Build vocabulary from labels
    char2idx = build_vocab(filenames)
    max_len = max(len(label) for label in labels)

    # Encode and pad labels
    encoded_labels = [
        [char2idx[c] for c in label] for label in labels
    ]
    padded_labels = tf.keras.preprocessing.sequence.pad_sequences(
        encoded_labels, maxlen=max_len, padding='post'
    )

    return images, padded_labels, char2idx

@nvtx.annotate("Training Model", color="green")
def train():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()

    train_path = '/home/stud3/Desktop/test_model_training/test_training/basic_CNN/dataset/train'
    val_path = '/home/stud3/Desktop/test_model_training/test_training/basic_CNN/dataset/val'
    test_path = '/home/stud3/Desktop/test_model_training/test_training/basic_CNN/dataset/test'

    # Load training data
    train_images, train_labels, char2idx = load_dataset_from_filenames(train_path)

    # Load val and test — reuse char2idx to ensure consistent label encoding
    val_images, val_labels, _ = load_dataset_from_filenames(val_path)
    test_images, test_labels, _ = load_dataset_from_filenames(test_path)

    # Encode validation and test labels with same vocab
    def encode_with_vocab(labels, char2idx):
        encoded = [[char2idx[c] for c in label] for label in labels]
        max_len = max(len(seq) for seq in encoded)
        return tf.keras.preprocessing.sequence.pad_sequences(encoded, maxlen=max_len, padding='post')

    val_labels = encode_with_vocab(val_labels, char2idx)
    test_labels = encode_with_vocab(test_labels, char2idx)

    # Convert to tf.data.Dataset
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(1000).batch(32)
    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32)

    print(f"Train size: {len(train_images)}")
    print(f"Val size: {len(val_images)}")
    print(f"Test size: {len(test_images)}")

    train_ds = tf.keras.utils.normalize(train_ds, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    with mlflow.start_run():
        mflow.log_param("epochs", 100)
        mlflow.log_param("batch_size",32)
        model.fit(trains_ds, 
                  epochs=100,
                  validation_data=val_ds,
                  callbacks=[MLflowLogger])

        loss, accurracy = model.evaluate(test_ds)
        mlflow.log_metric("test_loss", loss)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.tensorflow.log_model(model, "model")

    # model.fit(x_train, y_train, epochs=100)
    model.summary()
    model.save('/home/stud3/Desktop/test_model_training/test_training/basic_CNN/plate_CNN_01.keras')

    evaluate_model(x_test, y_test)

@nvtx.annotate("Evaluating Trained Model", color="pink")
def evaluate_model(x_test, y_test):
    model= tf.keras.models.load_model('/home/stud3/Desktop/test_model_training/test_training/basic_CNN/plate_CNN_01.keras')
    loss, accuracy = model.evaluate(x_test, y_test)

    print(loss)
    print(accuracy)

@nvtx.annotate("Inference", color="yellow")
def apply(model):
    image_number = 1
    while os.path.isfile(f"/home/stud3/Desktop/test_model_training/test_training/digits/digit{image_number}.png"):
        try:
            img = cv2.imread(f"/home/stud3/Desktop/test_model_training/test_training/digits/digit{image_number}.png")[:,:,0]
            img = np.invert(np.array([img]))
            prediction = model.predict(img)
            print(f"The number is probably a {np.argmax(prediction)}")
            plt.imshow(img[0], cmap=plt.cm.binary)
            plt.show()
        except Exception as e:
            print(f"Error: {e}")
        finally:
            image_number += 1

if __name__ == '__main__':
    train()
    model = tf.keras.models.load_model('/home/stud3/Desktop/test_model_training/test_training/handwritten.keras')
    #apply(model)