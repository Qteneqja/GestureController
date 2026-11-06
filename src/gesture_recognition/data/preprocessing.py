from pathlib import Path

import numpy as np
from PIL import Image


def load_sequence(video_folder, num_frames=16, image_size=(112, 112),
                  rotation_angle=0.0):
    """Load a video folder as a sequence of normalized RGB frames."""
    video_folder = Path(video_folder)
    frame_paths = sorted(video_folder.glob("*.jpg"))

    if not frame_paths:
        raise ValueError(f"No JPG frames found in {video_folder}")

    # Sample frames evenly from the beginning to the end.
    indices = np.linspace(0, len(frame_paths) - 1, num_frames).astype(int)

    frames = []
    for index in indices:
        with Image.open(frame_paths[index]) as image:
            image = image.convert("RGB").resize(image_size)

            if rotation_angle:
                image = image.rotate(
                    rotation_angle, resample=Image.Resampling.BILINEAR
                )

            # Convert pixel values from 0–255 to 0–1.
            pixels = np.asarray(image, dtype=np.float32) / 255.0
            frames.append(pixels)

    return np.stack(frames)

