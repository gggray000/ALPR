import tensorflow as tf
try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except: pass
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from mltu.tensorflow.dataProvider import DataProvider
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
from mltu.tensorflow.metrics import CWERMetric
from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding
from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate
from mltu.annotations.images import CVImage # Ensure this is imported

from model import train_model
from configs import ModelConfigs
import os
import cv2
import numpy as np

class GrayscaleConverter:
    def __call__(self, image_input, annotation):
        image_array = None

        if isinstance(image_input, CVImage):
            if hasattr(image_input, 'data') and isinstance(image_input.data, np.ndarray):
                image_array = image_input.data
            elif hasattr(image_input, 'image') and isinstance(image_input.image, np.ndarray):
                image_array = image_input.image
            else:
                 try:
                    _ = image_input.shape
                    _ = image_input.ndim
                    _ = image_input.size
                    image_array = image_input
                 except AttributeError:
                    raise TypeError(
                        f"CVImage object received but could not extract valid numpy array. "
                        f"Expected .data or .image attribute, or CVImage to behave as an array itself. "
                        f"CVImage attributes: {dir(image_input)}"
                    )

        elif isinstance(image_input, str):
            image_array = cv2.imread(image_input)
            if image_array is None:
                raise ValueError(f"Could not read image from path: {image_input}. Check path and integrity.")
        elif isinstance(image_input, np.ndarray):
            image_array = image_input
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}. Expected string, numpy array, or CVImage.")

        if not isinstance(image_array, np.ndarray) or image_array.size == 0:
            raise ValueError(f"GrayscaleConverter received invalid or empty image data: {type(image_array)}")

        # Ensure the image has appropriate dimensions for color conversion
        if image_array.ndim == 2:
            # Grayscale (H, W) -> BGR (H, W, 3)
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
        elif image_array.ndim == 3:
            if image_array.shape[-1] == 1:
                # Grayscale (H, W, 1) -> BGR (H, W, 3)
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
            elif image_array.shape[-1] == 4:
                # RGBA (H, W, 4) -> BGR (H, W, 3)
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
            elif image_array.shape[-1] != 3:
                raise ValueError(f"Unsupported number of channels: {image_array.shape[-1]}. Expected 1, 3, or 4.")
            # If 3 channels, assume it's BGR and proceed.

        # Convert to grayscale (H, W)
        gray_image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

        # *** CRUCIAL CHANGE HERE: Convert grayscale to a 3-channel (pseudo-BGR) image ***
        # This creates a 3-channel image where all channels are identical (grayscale).
        # This allows augmentors like RandomBrightness that expect 3 channels to function.
        final_image_array = cv2.cvtColor(gray_image_array, cv2.COLOR_GRAY2BGR)

        # Wrap the processed NumPy array back into a CVImage object
        return CVImage(final_image_array), annotation


class CustomRandomRotate:
    def __init__(self, max_angle=3):
        self.max_angle = max_angle

    def __call__(self, image_input):
        if isinstance(image_input, CVImage):
            if hasattr(image_input, 'data') and isinstance(image_input.data, np.ndarray):
                image_array = image_input.data
            elif hasattr(image_input, 'image') and isinstance(image_input.image, np.ndarray):
                image_array = image_input.image
            else:
                 try:
                    _ = image_input.shape
                    _ = image_input.ndim
                    _ = image_input.size
                    image_array = image_input
                 except AttributeError:
                    raise TypeError(
                        f"CustomRandomRotate received CVImage, but could not extract valid numpy array. "
                        f"CVImage attributes: {dir(image_input)}"
                    )
        elif isinstance(image_input, np.ndarray):
            image_array = image_input
        else:
            raise TypeError(f"Expected image to be a numpy array or CVImage, but got {type(image_input)}")

        if image_array.ndim not in [2, 3]:
            raise ValueError(f"Invalid image dimensions for rotation: {image_array.shape}. Expected 2D or 3D.")

        angle = np.random.uniform(-self.max_angle, self.max_angle)
        h, w = image_array.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        rotated = cv2.warpAffine(image_array, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # Since GrayscaleConverter now outputs 3 channels, this check is potentially simplified.
        # However, keep it if other augmentors might process differently, or if RandomErodeDilate
        # also takes a single channel, as it might convert back.
        # For now, it's safer to assume it's 3-channel and keep it that way for augmentors.
        # if len(image_array.shape) == 3 and image_array.shape[-1] == 1:
        #     rotated = np.expand_dims(rotated, axis=-1)

        return CVImage(rotated)


# Create a list of all the images and labels in the dataset
dataset, vocab, max_len = [], set(), 0
dataset_path = os.path.join("..", "Datasets", "plates")
for file in os.listdir(dataset_path):
    if file.endswith(".jpg") or file.endswith(".png"):
        label = os.path.splitext(file)[0]
        file_path = os.path.join(dataset_path, file)
        dataset.append([file_path, label])
        vocab.update(label)
        max_len = max(max_len, len(label))
# Save vocab and maximum text length to configs
configs = ModelConfigs()
configs.vocab = "".join(sorted(vocab))
configs.max_text_length = max_len
configs.save()


#Custom ImageReader for RGB license plate images
class ColorImageReader:
    def __call__(self, sample):
        image_path, label = sample
        import cv2
        image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return image, label


# Create a data provider for the dataset
data_provider = DataProvider(
    dataset=dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CVImage),GrayscaleConverter()],
    #data_preprocessors=[ImageReader(CVImage),GrayscaleConverter()],
    transformers=[
        ImageResizer(configs.width, configs.height),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab))
    ]
)

# Split the dataset into training and validation sets
train_data_provider, val_data_provider = data_provider.split(split=0.9)

# Augment training data with random brightness, rotation and erode/dilate
train_data_provider.augmentors = [
    RandomBrightness(), # Now expects a 3-channel image and should work
    CustomRandomRotate(max_angle=3), # Now receives a 3-channel image
    RandomErodeDilate() # Should also be fine with 3-channel or convert internally
]

# Creating TensorFlow model architecture
model = train_model(
    input_dim=(configs.height, configs.width, 3),  # *** IMPORTANT: Change input_dim to 3 channels ***
    output_dim=len(configs.vocab)
)

# Compile the model and print summary
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=configs.learning_rate),
    loss=CTCloss(),
    metrics=[CWERMetric(padding_token=len(configs.vocab))],
    run_eagerly=False
)

model.summary(line_length=110)

os.makedirs(configs.model_path, exist_ok=True)

# Define callbacks
callbacks = [
    EarlyStopping(monitor="val_CER", patience=50, verbose=1, mode="min"),
    ModelCheckpoint(f"{configs.model_path}/model.h5", monitor="val_CER", verbose=1, save_best_only=True, mode="min"),
    TrainLogger(configs.model_path),
    TensorBoard(f"{configs.model_path}/logs", update_freq=1),
    ReduceLROnPlateau(monitor="val_CER", factor=0.9, patience=20, min_delta=1e-10, verbose=1, mode="min"),
    Model2onnx(f"{configs.model_path}/model.h5"),
]

# Train the model
model.fit(
    train_data_provider,
    validation_data=val_data_provider,
    epochs=configs.train_epochs,
    callbacks=callbacks,
    workers=configs.train_workers
)

# Save training and validation datasets as csv files
train_data_provider.to_csv(os.path.join(configs.model_path, "train.csv"))
val_data_provider.to_csv(os.path.join(configs.model_path, "val.csv"))