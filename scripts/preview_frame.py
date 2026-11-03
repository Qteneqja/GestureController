from pathlib import Path
from PIL import Image

import numpy as np

# Choose the first frame from video folder 1.
image_path = Path(
    r"C:\JesterData\frames\20bn-jester-v1\1\00001.jpg"
)

with Image.open(image_path) as image:
    print("Original size:", image.size)

    # Use three colour channels and a consistent image size.
    resized = image.convert("RGB").resize((112, 112))
    print("Resized size:", resized.size)
    print("Colour mode:", resized.mode)

# Convert the resized image to decimal numbers and scale to 0–1.
pixels = np.asarray(resized, dtype=np.float32) / 255.0

print("Image shape:", pixels.shape)
print("Data type:", pixels.dtype)
print("Pixel range:", pixels.min(), "to", pixels.max())    