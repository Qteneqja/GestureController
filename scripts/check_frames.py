import csv
from pathlib import Path

# Find our project folder and prepared label files.
project_root = Path(__file__).resolve().parent.parent
labels_folder = project_root / "data/processed"

# Location of the extracted video folders on your computer.
frames_folder = Path(r"C:\JesterData\frames\20bn-jester-v1")

print("Image folder exists:", frames_folder.is_dir())

# Check that each listed video has at least one JPG frame.
for split in ["train", "validation", "test"]:
    checked = 0
    missing = 0

    with (labels_folder / f"{split}.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        for row in csv.DictReader(file):
            video_folder = frames_folder / row["video_id"]
            checked += 1

            if not any(video_folder.glob("*.jpg")):
                missing += 1

    print(f"{split}: {checked} videos checked, {missing} without JPG frames")