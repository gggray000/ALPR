import os
from PIL import Image, ImageDraw, ImageFont
import random
from tqdm import tqdm

# Configuration
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
FONTS_DIR = r"/fonts"  # contains .ttf files
OUTPUT_DIR = r"/dataset"
IMG_SIZE = (32, 32)
NUM_IMAGES_PER_CHAR = 300

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
for char in CHARS:
    os.makedirs(os.path.join(OUTPUT_DIR, char), exist_ok=True)

# Load fonts
fonts = []
for file in os.listdir(FONTS_DIR):
    if file.endswith(".ttf"):
        fonts.append(os.path.join(FONTS_DIR, file))

# Generate images
for char in tqdm(CHARS, desc="Generating characters"):
    for i in range(NUM_IMAGES_PER_CHAR):
        font_path = random.choice(fonts)
        font_size = random.randint(20, 28)
        font = ImageFont.truetype(font_path, font_size)

        img = Image.new("L", IMG_SIZE, color=255)
        draw = ImageDraw.Draw(img)

        # Random position to create diversity
        bbox = draw.textbbox((0, 0), char, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = (IMG_SIZE[0] - w) // 2 + random.randint(-2, 2)
        y = (IMG_SIZE[1] - h) // 2 + random.randint(-2, 2)

        draw.text((x, y), char, font=font, fill=random.randint(0, 50))

        # Optional: add light noise
        if random.random() < 0.3:
            for _ in range(10):
                rx, ry = random.randint(0, 31), random.randint(0, 31)
                img.putpixel((rx, ry), random.randint(0, 255))

        filename = f"{char}_{i:03d}.png"
        img.save(os.path.join(OUTPUT_DIR, char, filename))
