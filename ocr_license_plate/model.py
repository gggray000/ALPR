from keras import layers
from keras.models import Model
from mltu.tensorflow.model_utils import residual_block

def train_model(input_dim, output_dim, activation="leaky_relu", dropout=0.3):
    inputs = layers.Input(shape=input_dim, name="input")
    x = layers.Rescaling(1.0 / 255)(inputs)

    # CNN feature extractor
    for filters, stride, skip in [(16, 1, True), (16, 2, True), (16, 1, False),
                                   (32, 2, True), (32, 1, False),
                                   (64, 2, True), (32, 1, True),
                                   (64, 2, True), (64, 1, False)]:
        x = residual_block(x, filters, activation=activation, skip_conv=skip, strides=stride, dropout=dropout)
        x = layers.BatchNormalization()(x)

    # Reshape
    x = layers.Reshape((-1, x.shape[-1]))(x)

    # BiLSTM layers
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)

    # Output
    output = layers.Dense(output_dim + 1, activation="softmax", name="output")(x)

    return Model(inputs=inputs, outputs=output)
