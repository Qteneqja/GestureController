import csv
from collections import Counter
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

for split in ["train", "validation", "test"]:
    csv_path = project_root / "data" / "processed" / f"{split}.csv"

    with csv_path.open(encoding="utf-8", newline="") as file:
        counts = Counter(row["gesture"] for row in csv.DictReader(file))

    print(f"\n{split}: {sum(counts.values())} videos")
    for gesture, count in sorted(counts.items()):
        print(f"{gesture}: {count}")