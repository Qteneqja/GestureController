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