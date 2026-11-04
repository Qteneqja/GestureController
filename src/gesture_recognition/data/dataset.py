import csv
from pathlib import Path

import numpy as np

from .preprocessing import load_sequence


def load_dataset(csv_path, frames_folder, augment=False, seed=None):
    """Load one video and label at a time, with optional augmentation."""
    frames_folder = Path(frames_folder)
    rng = np.random.default_rng(seed)

    with open(csv_path, encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            video_folder = frames_folder / row["video_id"]
            sequence = load_sequence(video_folder)
            class_id = int(row["class_id"])

            if augment:
                # Apply the same brightness change to every frame.
                brightness = np.float32(rng.uniform(0.8, 1.2))
                sequence = np.clip(sequence * brightness, 0.0, 1.0)

            yield sequence, class_id

def make_tf_dataset(csv_path, frames_folder, batch_size=8, training=False):
    """Prepare batches of videos and labels for TensorFlow."""
    import tensorflow as tf

    dataset = tf.data.Dataset.from_generator(
        lambda: load_dataset(
            csv_path, frames_folder, augment=training
        ),
        output_signature=(
            tf.TensorSpec(shape=(16, 112, 112, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )

    # Shuffle training videos; keep evaluation order unchanged.
    if training:
        dataset = dataset.shuffle(buffer_size=32, seed=42)

    return dataset.batch(batch_size).prefetch(1)
