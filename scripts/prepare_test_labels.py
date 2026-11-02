"""Prepare held-out test labels from official Jester validation labels."""

import csv
import json
from collections import Counter
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
data_folder = project_root / "data/processed"
config_path = project_root / "config/gesture_classes.json"
input_path = project_root / "data/raw/jester_labels/validation.csv"

# Use the same class IDs as training and validation.
with config_path.open(encoding="utf-8") as file:
    config = json.load(file)

label_to_id = {
    item["label"]: item["id"]
    for item in config["classes"]
}

# Collect IDs already used for training or internal validation.
used_ids = set()

for filename in ["train.csv", "validation.csv"]:
    with (data_folder / filename).open(
        encoding="utf-8", newline=""
    ) as file:
        for row in csv.DictReader(file):
            used_ids.add(row["video_id"])

test_rows = []
test_ids = set()
class_counts = Counter()

with input_path.open(encoding="utf-8-sig", newline="") as file:
    for video_id, gesture in csv.reader(file, delimiter=";"):
        if gesture not in label_to_id:
            continue

        if video_id in used_ids:
            raise ValueError(f"Test video overlaps another split: {video_id}")

        if video_id in test_ids:
            raise ValueError(f"Duplicate test video: {video_id}")

        test_ids.add(video_id)
        test_rows.append([video_id, gesture, label_to_id[gesture]])
        class_counts[gesture] += 1

# Check that all seven classes are represented.
missing = set(label_to_id) - set(class_counts)
if missing:
    raise ValueError(f"Missing test classes: {sorted(missing)}")

output_path = data_folder / "test.csv"

with output_path.open("w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["video_id", "gesture", "class_id"])
    writer.writerows(test_rows)

print(f"Test videos: {len(test_rows)}")

for gesture in label_to_id:
    print(f"{gesture}: {class_counts[gesture]}")

print("Overlap with training and validation: 0 videos")