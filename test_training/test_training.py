import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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

    # what is optimizer and loss function?
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    model.save('handwritten.keras')

    # after training
    model= tf.keras.models.load_model('handwritten.keras')
    loss, accuracy = model.evaluate(x_test, y_test)

    print(loss)
    print(accuracy)

def apply(model):
    image_number = 1
    while os.path.isfile(f"digits/digit{image_number}.png"):
        try:
            img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
            img = np.invert(np.array([img]))
            prediction = model.predict(img)
            # Giving the number with the highest activation. What is activation?
            print(f"The number is probably a {np.argmax(prediction)}")
            plt.imshow(img[0], cmap=plt.cm.binary)
            plt.show()
        except Exception as e:
            print(f"Error: {e}")
        finally:
            image_number += 1

if __name__ == '__main__':
    model = tf.keras.models.load_model('handwritten.keras')
    apply(model)