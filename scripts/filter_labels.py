"""Filter Jester training labels using the shared class configuration."""

import csv
import json
from collections import Counter
from pathlib import Path

# Locate the project and its files.
project_root = Path(__file__).resolve().parent.parent
config_path = project_root / "config/gesture_classes.json"
input_path = project_root / "data/raw/jester_labels/train.csv"
output_path = project_root / "data/processed/selected_train.csv"

# Read the agreed gesture names and class IDs.
with config_path.open(encoding="utf-8") as file:
    config = json.load(file)

label_to_id = {
    item["label"]: item["id"]
    for item in config["classes"]
}

selected_rows = []
class_counts = Counter()

# Keep videos whose gesture appears in our configuration.
with input_path.open(encoding="utf-8-sig", newline="") as file:
    for video_id, gesture in csv.reader(file, delimiter=";"):
        if gesture in label_to_id:
            selected_rows.append(
                [video_id, gesture, label_to_id[gesture]]
            )
            class_counts[gesture] += 1

# Check that every selected class exists before saving.
missing = set(label_to_id) - set(class_counts)
if missing:
    raise ValueError(f"Selected classes missing from input: {sorted(missing)}")

# Save a new CSV without changing the original labels.
output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["video_id", "gesture", "class_id"])
    writer.writerows(selected_rows)

print(f"Selected training videos: {len(selected_rows)}")

for gesture in label_to_id:
    print(f"{gesture}: {class_counts[gesture]}")

print(f"\nSaved to: {output_path}")