"""Tests for the CNN weakness analysis.

Run from the project root:
    python -m pytest tests/test_analysis.py
"""

import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.gesture_recognition.training.analyze_cnn import (  # noqa: E402
    confusion_counts,
    load_class_names,
    summarize_errors,
)

CLASS_FILE = project_root / "config" / "gesture_classes.json"


def test_class_names_are_in_id_order():
    names = load_class_names(CLASS_FILE)
    assert names[0] == "Swiping Left"
    assert names[6] == "No gesture"
    assert len(names) == 7


def test_confusion_counts_rows_are_true_labels():
    matrix = confusion_counts([0, 0, 1], [0, 1, 1], num_classes=7)
    assert matrix[0, 0] == 1   # Swiping Left predicted correctly once
    assert matrix[0, 1] == 1   # Swiping Left predicted as Swiping Right once
    assert matrix[1, 1] == 1
    assert matrix.sum() == 3


def test_error_types_are_counted_separately():
    names = load_class_names(CLASS_FILE)
    y_true = [0, 0, 0, 2, 4, 5]
    y_pred = [0, 1, 2, 3, 4, 5]
    # 0 -> 1 is a direction error, 0 -> 2 is a hand-shape error,
    # 2 -> 3 is a direction error.
    summary = summarize_errors(confusion_counts(y_true, y_pred, 7), names)

    assert summary["total_errors"] == 3
    assert summary["direction_errors"] == 2
    assert summary["hand_shape_errors"] == 1
    assert summary["accuracy"] == round(3 / 6, 4)


def test_per_class_accuracy_and_most_confused():
    names = load_class_names(CLASS_FILE)
    y_true = [0, 0, 0, 0]
    y_pred = [0, 1, 1, 0]
    summary = summarize_errors(confusion_counts(y_true, y_pred, 7), names)
    swiping_left = summary["per_class"][0]

    assert swiping_left["accuracy"] == 0.5
    assert swiping_left["most_confused_with"] == "Swiping Right"
    assert swiping_left["confused_share"] == 0.5
    assert np.isclose(sum(r["videos"] for r in summary["per_class"]), 4)