"""Tests for the webcam app logic. No webcam, model or TensorFlow needed.

Run with:  pytest tests/test_application.py
"""

import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from gesture_recognition.application.gesture_controller import GestureController  # noqa: E402
from gesture_recognition.application.inference import (  # noqa: E402
    FrameBuffer,
    load_classes,
    preprocess_frame,
)

KEYS = {"next_slide": ["right"], "no_action": []}


def test_preprocess_matches_training_format():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :, 0] = 255  # pure blue in BGR
    out = preprocess_frame(frame, (112, 112))
    assert out.shape == (112, 112, 3)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    # Blue should end up in the last channel after BGR -> RGB.
    assert out[0, 0, 2] == 1.0 and out[0, 0, 0] == 0.0


def test_buffer_samples_16_frames_across_window():
    buffer = FrameBuffer(window_seconds=3.0, num_frames=16)
    for i in range(90):  # 3 seconds at 30 fps
        buffer.add(np.full((112, 112, 3), i, dtype=np.float32), i / 30)
    assert buffer.is_ready()
    sequence = buffer.get_sequence()
    assert sequence.shape == (16, 112, 112, 3)
    # First and last sampled frames come from the two ends of the window.
    assert sequence[0, 0, 0, 0] == buffer.frames[0][1][0, 0, 0]
    assert sequence[-1, 0, 0, 0] == 89


def test_buffer_not_ready_too_early():
    buffer = FrameBuffer(window_seconds=3.0, num_frames=16)
    for i in range(20):  # less than 1 second
        buffer.add(np.zeros((2, 2, 3), dtype=np.float32), i / 30)
    assert not buffer.is_ready()


def test_class_order_matches_shared_file():
    classes = load_classes(project_root / "config" / "gesture_classes.json")
    assert [c["id"] for c in classes] == list(range(len(classes)))


def test_disabled_controller_never_acts():
    c = GestureController(KEYS, start_enabled=False)
    assert c.handle("next_slide", 0.99, now=0) == (False, "disabled")


def test_low_confidence_is_ignored():
    c = GestureController(KEYS, confidence_threshold=0.8, start_enabled=True)
    assert c.handle("next_slide", 0.5, now=0) == (False, "low confidence")


def test_no_action_class_is_ignored():
    c = GestureController(KEYS, start_enabled=True)
    assert c.handle("no_action", 0.99, now=0) == (False, "no gesture")


def test_cooldown_blocks_repeat_then_allows():
    c = GestureController(KEYS, cooldown_seconds=1.5, start_enabled=True)
    assert c.handle("next_slide", 0.99, now=10.0)[0] is True
    assert c.handle("next_slide", 0.99, now=10.5) == (False, "cooldown")
    assert c.handle("next_slide", 0.99, now=12.0)[0] is True


def test_session_log_writes_rows_and_settings(tmp_path):
    import csv
    import json

    from gesture_recognition.application.session_log import SessionLog

    log = SessionLog(tmp_path, "swipe right x5", {"window_seconds": 1.5},
                     start_time=100.0)
    log.write(101.0, "model", "Swiping Right", "next_slide", 0.91,
              True, True, "ok", 72.4)
    log.write(102.0, "simulated", "Thumb Up", "start_presentation", 1.0,
              True, False, "cooldown")
    log.close()

    with open(log.path, encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert rows[0]["label"] == "Swiping Right" and rows[0]["fired"] == "1"
    assert rows[0]["time_s"] == "1.0"
    assert rows[1]["model_ms"] == ""
    assert " " not in log.path.name  # spaces replaced in the file name

    info = json.loads(log.path.with_suffix(".json").read_text(encoding="utf-8"))
    assert info["window_seconds"] == 1.5 and info["session"] == "swipe right x5"
