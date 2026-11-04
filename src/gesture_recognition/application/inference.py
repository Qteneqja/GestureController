"""Turn live webcam frames into model predictions.

Three parts:
1. preprocess_frame: make one webcam frame look like a training frame.
2. FrameBuffer: remember the last few seconds of frames and sample 16.
3. GesturePredictor: load the model and the class list, then predict.
"""

import json
from collections import deque

import numpy as np
from PIL import Image


def preprocess_frame(frame_bgr, image_size=(112, 112)):
    """Convert one OpenCV frame into the same format used in training.

    These steps copy src/gesture_recognition/data/preprocessing.py
    (load_sequence): RGB, Pillow resize, float32 scaled to 0..1.
    Using Pillow for the resize (not cv2.resize) keeps the pixels as
    close as possible to what the model saw during training.
    """
    frame_rgb = frame_bgr[:, :, ::-1]  # OpenCV gives BGR, the model expects RGB
    image = Image.fromarray(np.ascontiguousarray(frame_rgb)).resize(tuple(image_size))
    return np.asarray(image, dtype=np.float32) / 255.0


class FrameBuffer:
    """Keeps the most recent `window_seconds` of processed frames."""

    def __init__(self, window_seconds=3.0, num_frames=16):
        self.window_seconds = window_seconds
        self.num_frames = num_frames
        self.frames = deque()  # items are (timestamp, frame)

    def add(self, frame, timestamp):
        self.frames.append((timestamp, frame))
        # Drop frames older than the window.
        while self.frames and timestamp - self.frames[0][0] > self.window_seconds:
            self.frames.popleft()

    def clear(self):
        self.frames.clear()

    def is_ready(self, min_fill=0.9):
        """True once the buffer covers most of the time window."""
        if len(self.frames) < self.num_frames:
            return False
        span = self.frames[-1][0] - self.frames[0][0]
        return span >= self.window_seconds * min_fill

    def get_sequence(self):
        """Sample `num_frames` evenly from start to end of the window.

        Same idea as np.linspace in load_sequence, so a 3 second gesture
        is spread over 16 frames the way the training clips were.
        Returns shape (num_frames, height, width, 3).
        """
        indices = np.linspace(0, len(self.frames) - 1, self.num_frames).astype(int)
        return np.stack([self.frames[i][1] for i in indices])


def load_classes(class_mapping_path):
    """Read the shared class list, ordered by id (the model's output order)."""
    with open(class_mapping_path, encoding="utf-8") as file:
        classes = json.load(file)["classes"]
    return sorted(classes, key=lambda item: item["id"])


class GesturePredictor:
    def __init__(self, model_path, class_mapping_path):
        import keras  # imported here so tests can run without loading TensorFlow

        self.model = keras.models.load_model(model_path)
        self.classes = load_classes(class_mapping_path)

        outputs = self.model.output_shape[-1]
        if outputs != len(self.classes):
            raise ValueError(
                f"Model has {outputs} outputs but the class file has "
                f"{len(self.classes)} classes. They must match."
            )

    def predict(self, sequence):
        """Return (label, command, confidence) for one 16-frame sequence."""
        batch = sequence[np.newaxis, ...]  # add batch dimension -> (1, 16, H, W, 3)
        probabilities = self.model.predict(batch, verbose=0)[0]
        best = int(np.argmax(probabilities))
        item = self.classes[best]
        return item["label"], item["command"], float(probabilities[best])
