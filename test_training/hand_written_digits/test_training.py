import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import nvtx
import mlflow

mlflow.set_tracking_uri(uri="http://localhost:5001")
mlflow.set_experiment("hand_written_digits")

class MLflowLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for key, value in logs.items():
                mlflow.log_metric(key, value, step=epoch)

class NvtxEpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_range = nvtx.start_range(f"Epoch {epoch+1}", color="blue")

    def on_epoch_end(self, epoch, logs=None):
        nvtx.end_range(self.epoch_range)
        print(f"[NVTX] Epoch {epoch+1} finished.")

@nvtx.annotate("Normalizing Data", color="blue")
def normalize_data(x_train, x_test):
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)
    return x_train, x_test

@nvtx.annotate("Training Model", color="green")
def train():
    mnist = tf.keras.datasets.mnist
    (x_train_raw, y_train),(x_test_raw, y_test) = mnist.load_data()

    x_train, x_test = normalize_data(x_train_raw, x_test_raw)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    with mlflow.start_run():
        mlflow.log_param("epochs", 200)
        mlflow.log_param("batch_size",32)
        model.fit(x_train,
                  y_train, 
                  epochs=200,
                  callbacks=[MLflowLogger(), NvtxEpochLogger()])

        loss, accurracy = model.evaluate(x_test, y_test)
        mlflow.log_metric("test_loss", loss)
        mlflow.log_metric("test_accuracy", accurracy)
        mlflow.tensorflow.log_model(model, "model")

    model.summary()
    model.save(f"/home/stud3/Desktop/ALPR/test_training/hand_written_digits/handwritten.keras")

    evaluate_model(x_test, y_test)

@nvtx.annotate("Evaluating Trained Model", color="pink")
def evaluate_model(x_test, y_test):
    model= tf.keras.models.load_model(f"/home/stud3/Desktop/ALPR/test_training/hand_written_digits/handwritten.keras")
    loss, accuracy = model.evaluate(x_test, y_test)

    print(loss)
    print(accuracy)

@nvtx.annotate("Single Inference", color="yellow")
def infer_single_image(img_path, model):
    img = cv2.imread(img_path)[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f"The number in {img_path} is probably a {np.argmax(prediction)}")
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()

def apply(model):
    image_number = 1
    while os.path.isfile(f"/home/stud3/Desktop/ALPR/test_training/hand_written_digits/digits/digit{image_number}.png"):
        try:
            infer_single_image(f"/home/stud3/Desktop/ALPR/test_training/hand_written_digits/digits/digit{image_number}.png", model)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            image_number += 1

if __name__ == '__main__':
    train()
    model = tf.keras.models.load_model(f"/home/stud3/Desktop/ALPR/test_training/hand_written_digits/handwritten.keras")
    apply(model)