"""Create reproducible training and validation splits by gesture class."""

import csv
import random
from collections import defaultdict
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
data_folder = project_root / "data/processed"
input_path = data_folder / "selected_train.csv"

# Fixed settings make the split reproducible.
validation_fraction = 0.20
rng = random.Random(42)

# Group videos by class and check for duplicate video IDs.
videos_by_class = defaultdict(list)
seen_ids = set()

with input_path.open(encoding="utf-8", newline="") as file:
    for row in csv.DictReader(file):
        if row["video_id"] in seen_ids:
            raise ValueError(f"Duplicate video ID: {row['video_id']}")

        seen_ids.add(row["video_id"])
        videos_by_class[int(row["class_id"])].append(row)

train_rows = []
validation_rows = []

# Split each class separately to preserve class proportions.
for class_id in sorted(videos_by_class):
    rows = sorted(
        videos_by_class[class_id],
        key=lambda row: int(row["video_id"])
    )
    rng.shuffle(rows)

    validation_count = round(len(rows) * validation_fraction)
    validation_rows.extend(rows[:validation_count])
    train_rows.extend(rows[validation_count:])

# Mix the classes within each output.
rng.shuffle(train_rows)
rng.shuffle(validation_rows)

# Each video belongs to only one split.
train_ids = {row["video_id"] for row in train_rows}
validation_ids = {row["video_id"] for row in validation_rows}

if train_ids & validation_ids:
    raise ValueError("A video appears in both splits.")

# Save the two label files.
for filename, rows in [
    ("train.csv", train_rows),
    ("validation.csv", validation_rows),
]:
    output_path = data_folder / filename

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["video_id", "gesture", "class_id"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"{filename}: {len(rows)} videos")

print("Overlap between training and validation: 0 videos")
print("Random seed: 42")