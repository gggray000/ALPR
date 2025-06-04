import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import nvtx
#import mlflow

#mlflow.set_tracking_uri(uri="http://localhost:5000")
#mlflow.set_experiment("hand_written_digits")

class MLflowLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for key, value in logs.item():
                mlfow.log_metric(key, value, step=epoch)

@nvtx.annotate("Training Model", color="green")
def train():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # with mlflow.start_run():
    #     mlflow.log_param("epochs", 300)
    #     mlflow.log_param("batch_size",32)
    #     model.fit(trains_ds, 
    #               epochs=100,
    #               validation_data=val_ds,
    #               callbacks=[MLflowLogger])

    #     loss, accurracy = model.evaluate(test_ds)
    #     mlflow.log_metric("test_loss", loss)
    #     mlflow.log_metric("test_accuracy", accuracy)
    #     mlflow.tensorflow.log_model(model, "model")

    model.fit(x_train, y_train, epochs=500)
    model.summary()
    model.save(f"/home/stud3/Desktop/ALPR/test_training/hand_written_digits/handwritten.keras")

    evaluate_model(x_test, y_test)

@nvtx.annotate("Evaluating Trained Model", color="pink")
def evaluate_model(x_test, y_test):
    model= tf.keras.models.load_model(f"/home/stud3/Desktop/ALPR/test_training/hand_written_digits/handwritten.keras")
    loss, accuracy = model.evaluate(x_test, y_test)

    print(loss)
    print(accuracy)

@nvtx.annotate("Inference", color="yellow")
def apply(model):
    image_number = 1
    while os.path.isfile(f"/home/stud3/Desktop/ALPR/test_training/hand_written_digits/digits/digit{image_number}.png"):
        try:
            img = cv2.imread(f"/home/stud3/Desktop/ALPR/test_training/hand_written_digits/digits/digit{image_number}.png")[:,:,0]
            img = np.invert(np.array([img]))
            prediction = model.predict(img)
            print(f"The number in file No.{image_number} is probably a {np.argmax(prediction)}")
            plt.imshow(img[0], cmap=plt.cm.binary)
            plt.show()
        except Exception as e:
            print(f"Error: {e}")
        finally:
            image_number += 1

if __name__ == '__main__':
    #train()
    model = tf.keras.models.load_model(f"/home/stud3/Desktop/ALPR/test_training/hand_written_digits/handwritten.keras")
    apply(model)