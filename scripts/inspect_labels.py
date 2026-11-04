"""Count training videos for each Jester gesture class."""

import csv
from collections import Counter
from pathlib import Path

# Find the project folder using this script's location.
project_root = Path(__file__).resolve().parent.parent
labels_path = project_root / "data/raw/jester_labels/train.csv"

# Each row contains: video ID;gesture name
class_counts = Counter()

with labels_path.open(encoding="utf-8-sig", newline="") as file:
    reader = csv.reader(file, delimiter=";")

    for video_id, gesture in reader:
        class_counts[gesture] += 1

# Display the total and the number of videos per gesture.
print(f"Total training videos: {sum(class_counts.values())}")
print(f"Number of gesture classes: {len(class_counts)}")
print("\nTraining videos per class:")

for gesture, count in sorted(class_counts.items()):
    print(f"{gesture}: {count}")