from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from tqdm.auto import tqdm
from urllib.request import urlretrieve
from zipfile import ZipFile
import shutil
import matplotlib.pyplot as plt
import torch
import os
import glob

# cache_dir = os.path.expanduser("~/.cache/huggingface")
# shutil.rmtree(cache_dir)

device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")

def read_image(image_path):
    """
    :param image_path: String, path to the input image.

    Returns:
        image: PIL image
    """
    image = Image.open(image_path).convert("RGB")
    return image

def ocr(image, processor, model):
    """
    :param image: PIL image
    :param processor: Huggingface OCR processor
    :param model: Huggingface OCR model

    Returns:
        generated_text: the OCR'd text string
    """
    # Can directly perorm OCR on cropped images
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def eval_new_data(data_path=None, num_sample=4, model=None):
    image_paths = glob.glob(data_path)  # This returns a list of image file paths
    for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
        if i == num_sample:
            break
        image = read_image(image_path)
        text = ocr(image, processor, model)
        plt.figure(figsize=(7, 4))
        plt.imshow(image)
        plt.title(text)
        plt.axis('off')
        plt.show()

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed", use_fast=False)
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed").to(device)
eval_new_data(
    data_path = os.path.join('images', 'dataset_final', 'test', '*'),
    num_sample = 20,
    model = model
)