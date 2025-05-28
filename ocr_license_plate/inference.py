import cv2
import typing
import numpy as np

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer


class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        # --- Preprocessing steps to match training data ---
        # 1. Ensure the image is in the correct 3-channel grayscale format
        # The input 'image' here is a NumPy array from cv2.imread, usually (H, W, 3) BGR.

        image_array = image  # Start with the input image

        # Handle different channel configurations before conversion to grayscale
        if image_array.ndim == 2:
            # Grayscale (H, W) -> Convert to BGR (H, W, 3) temporarily for cv2.cvtColor(BGR2GRAY)
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
        elif image_array.ndim == 3:
            if image_array.shape[-1] == 1:
                # Grayscale (H, W, 1) -> Convert to BGR (H, W, 3)
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
            elif image_array.shape[-1] == 4:
                # RGBA (H, W, 4) -> Convert to BGR (H, W, 3)
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
            elif image_array.shape[-1] != 3:
                raise ValueError(
                    f"Unsupported number of channels in input image for prediction: {image_array.shape[-1]}. Expected 1, 3, or 4.")
            # If 3 channels, it's assumed to be BGR already and remains 'image_array'.
        else:
            raise ValueError(f"Unsupported image dimensions for prediction: {image_array.ndim}. Expected 2D or 3D.")

        # Convert the (now BGR) image to grayscale (H, W)
        gray_image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

        # Convert the grayscale image (H, W) back to a 3-channel (pseudo-BGR) format (H, W, 3)
        # This matches the input_dim=(H, W, 3) that your model was trained on
        processed_image = cv2.cvtColor(gray_image_array, cv2.COLOR_GRAY2BGR)

        # 2. Resize the processed image to the model's expected input dimensions
        # self.input_shapes[0][1:3][::-1] extracts (width, height) from the ONNX model's input shape
        processed_image = cv2.resize(processed_image, self.input_shapes[0][1:3][::-1])

        # 3. Add batch dimension (1, H, W, 3) and convert to float32
        image_pred = np.expand_dims(processed_image, axis=0).astype(np.float32)

        # Run inference
        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]

        # Decode the prediction using CTC decoder
        text = ctc_decoder(preds, self.char_list)[0]

        return text


if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs

    configs = BaseModelConfigs.load("/home/stud3/Desktop/ALPR/Models/license_plate_ocr/202505281503/configs.yaml")

    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

    df = pd.read_csv("/home/stud3/Desktop/ALPR/Models/license_plate_ocr/202505281503/val.csv").values.tolist()

    accum_cer = []
    for image_path, label in tqdm(df):
        image = cv2.imread(image_path.replace("\\", "/"))

        prediction_text = model.predict(image)

        cer = get_cer(prediction_text, label)
        print(f"Image: {image_path}, Label: {label}, Prediction: {prediction_text}, CER: {cer}")

        accum_cer.append(cer)

    print(f"Average CER: {np.average(accum_cer)}")