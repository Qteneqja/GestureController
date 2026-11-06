import csv
from pathlib import Path

import numpy as np

from .preprocessing import load_sequence


def load_dataset(csv_path, frames_folder, augment=False, seed=None,
                 num_frames=16, image_size=(112, 112)):
    """Load one video and its class label at a time."""
    frames_folder = Path(frames_folder)
    rng = np.random.default_rng(seed)

    with open(csv_path, encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            video_folder = frames_folder / row["video_id"]
            rotation_angle = rng.uniform(-5.0, 5.0) if augment else 0.0
            sequence = load_sequence(
                video_folder,
                num_frames=num_frames,
                image_size=image_size,
                rotation_angle=rotation_angle,
            )
            class_id = int(row["class_id"])

            if augment:
                # Keep brightness consistent across the video.
                brightness = np.float32(rng.uniform(0.8, 1.2))
                sequence = np.clip(sequence * brightness, 0.0, 1.0)

            yield sequence, class_id


def make_tf_dataset(csv_path, frames_folder, batch_size=8, training=False,
                    num_frames=16, image_size=(112, 112)):
    """Prepare video batches for TensorFlow."""
    import tensorflow as tf

    dataset = tf.data.Dataset.from_generator(
        lambda: load_dataset(
            csv_path,
            frames_folder,
            augment=training,
            num_frames=num_frames,
            image_size=image_size,
        ),
        output_signature=(
            tf.TensorSpec(
                shape=(num_frames, image_size[1], image_size[0], 3),
                dtype=tf.float32,
            ),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )

    if training:
        dataset = dataset.shuffle(buffer_size=32, seed=42)

    return dataset.batch(batch_size).prefetch(1)


def dataset_from_config(split, config_path="configs/data.yaml"):
    """Build a dataset using our shared settings."""
    import yaml

    if split not in ["train", "validation", "test"]:
        raise ValueError("Choose train, validation or test")

    with open(config_path, encoding="utf-8") as file:
        settings = yaml.safe_load(file)

    return make_tf_dataset(
        csv_path=settings[f"{split}_csv"],
        frames_folder=settings["frames_folder"],
        batch_size=settings["batch_size"],
        training=(split == "train"),
        num_frames=settings["num_frames"],
        image_size=tuple(settings["image_size"]),
    )