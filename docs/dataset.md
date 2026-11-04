# Jester Dataset Preparation

## Local folder structure

Extract the downloaded label files into:

data/raw/jester_labels/

Required for the current inspection script:
- train.csv

Raw data stays local and is excluded from Git.

## Inspect training labels

Run from the project folder:

python scripts/inspect_labels.py

Each CSV row contains a video ID and gesture name separated
by a semicolon. The file has no header row.

## Initial findings

- Training videos listed: 118,562
- Gesture classes: 27
- Doing other things: 9,592 videos
- Turning Hand Clockwise: 3,216 videos
- Most remaining classes contain roughly 4,000–4,400 videos.

These counts describe the label file. Image files have not
yet been checked.

## Next steps

- Finalize gesture classes and command mappings with the team.
- Prepare reproducible training, validation, and test splits.
- Implement shared frame preprocessing and dataset loaders.

## Dataset splits

We selected seven gesture classes.

- Training: 23,786 videos
- Validation: 5,946 videos
- Test: 3,623 videos

We split the selected official training data into 80% training
and 20% validation within each class, using random seed 42.

The selected official Jester validation data serves as our test set.
No video IDs overlap between the three sets.

These counts are based on labels. Image files still need checking.

## Shared dataset loader

Run Python from the GestureController project root.

```python
from src.gesture_recognition.data.dataset import make_tf_dataset

# Change this path to your local extracted dataset folder.
frames_folder = r"C:\JesterData\frames\20bn-jester-v1"

train_ds = make_tf_dataset(
    "data/processed/train.csv", frames_folder, training=True
)
validation_ds = make_tf_dataset(
    "data/processed/validation.csv", frames_folder, training=False
)
test_ds = make_tf_dataset(
    "data/processed/test.csv", frames_folder, training=False
)
```

Each video contains 16 evenly sampled RGB frames resized to
112 × 112. Pixel values are float32 between 0 and 1.
Short clips repeat frames to reach 16 frames.

The default batch size is 8. Video batches have shape
(8, 16, 112, 112, 3), with one integer class ID per video.
The final batch may contain fewer than 8 videos.

Training uses shuffling and random brightness factors from
0.8 to 1.2. Each video receives the same brightness factor
across all its frames. Validation and test use no augmentation
or shuffling.

Both models should use the same splits and class mapping.
The CNN baseline member handles frame-to-video predictions.

Required packages: NumPy, Pillow and TensorFlow.
Tested locally with TensorFlow 2.21.0.
One batch from each split loaded successfully.

Dataset images and generated CSVs are excluded from Git.
Teammates must obtain the images and regenerate the CSVs
using the preparation scripts.

## Recreate the split CSV files

Place the extracted Jester label files in:
data/raw/jester_labels/

This folder should contain labels.csv, train.csv,
validation.csv, test.csv and test-answers.csv.

From the project root, run these commands in order:

```powershell
python scripts/filter_labels.py
python scripts/create_splits.py
python scripts/prepare_test_labels.py
```

These create the CSV files used by the loader:
- data/processed/train.csv
- data/processed/validation.csv
- data/processed/test.csv

Expected counts:
- Training: 23,786 videos
- Validation: 5,946 videos
- Test: 3,623 videos

The official Jester training subset is split into training
and validation within each class, using random seed 42.
The selected official Jester validation subset is our test set.
No video IDs overlap between these splits.

Class IDs and computer commands are defined in:
config/gesture_classes.json