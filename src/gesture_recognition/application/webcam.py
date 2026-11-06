"""Webcam capture.

Wraps OpenCV so the rest of the app only deals with "give me the next
frame and the time it was taken".
"""

import time

import cv2


class Webcam:
    def __init__(self, camera_index=0):
        self.capture = cv2.VideoCapture(camera_index)
        if not self.capture.isOpened():
            raise RuntimeError(
                f"Could not open camera {camera_index}. "
                "Check that no other app is using it."
            )

    def read(self):
        """Return (frame, timestamp). Frame is BGR, which is OpenCV's default."""
        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError("Could not read a frame from the camera.")
        return frame, time.monotonic()

    def release(self):
        self.capture.release()
