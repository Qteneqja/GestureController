import csv
from pathlib import Path

from .preprocessing import load_sequence


def load_dataset(csv_path, frames_folder):
    """Read videos and their class IDs one at a time."""
    frames_folder = Path(frames_folder)

    with open(csv_path, encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            video_folder = frames_folder / row["video_id"]

            sequence = load_sequence(video_folder)
            class_id = int(row["class_id"])

            # Give one video and its label to the caller.
            yield sequence, class_id