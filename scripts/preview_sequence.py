from pathlib import Path
import numpy as np

from PIL import Image

video_folder = Path(r"C:\JesterData\frames\20bn-jester-v1\1")

# Sort frames into their original time order.
frame_paths = sorted(video_folder.glob("*.jpg"))

if not frame_paths:
    raise ValueError("No JPG frames found in the video folder.")

# Select 16 evenly spaced positions, including the first and last.
indices = np.linspace(0, len(frame_paths) - 1, 16).astype(int)
selected_paths = [frame_paths[index] for index in indices]

print("Available frames:", len(frame_paths))
print("Selected frames:", len(selected_paths))
print("Selected filenames:", [path.name for path in selected_paths])

# Resize and normalize each selected frame.
frames = []

for path in selected_paths:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((112, 112))
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        frames.append(pixels)

# Combine the frames into one video sequence.
sequence = np.stack(frames)

print("Sequence shape:", sequence.shape)
print("Data type:", sequence.dtype)
print("Pixel range:", sequence.min(), "to", sequence.max())