"""Main loop for the webcam demo.

Keys in the preview window:
  E      turn commands on or off
  1 to 7 pretend the model predicted that class (100% confidence).
         Useful for testing the safety rules before a real model exists.
  Q      quit

Every prediction is saved to a CSV log (see session_log.py).
"""

import time

import cv2
import yaml

from .gesture_controller import GestureController
from .inference import FrameBuffer, GesturePredictor, preprocess_frame
from .session_log import SessionLog
from .webcam import Webcam

# Settings copied into each log's JSON file, so every test is traceable.
LOGGED_SETTINGS = ["model_path", "window_seconds", "predict_every_seconds",
                   "confidence_threshold", "cooldown_seconds",
                   "rearm_seconds", "mirror_model_input", "dry_run"]


def load_config(path):
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def draw_overlay(frame, controller, status_lines):
    color = (0, 200, 0) if controller.enabled else (0, 0, 255)
    state = "ON" if controller.enabled else "OFF"
    mode = "DRY RUN" if controller.dry_run else "LIVE KEYS"
    cv2.putText(frame, f"Commands: {state} ({mode})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    for i, line in enumerate(status_lines):
        cv2.putText(frame, line, (10, 60 + 28 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "E: on/off   1-7: simulate   Q: quit",
                (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1)


def run(config_path, session_name="test"):
    config = load_config(config_path)
    image_size = tuple(config["image_size"])

    predictor = GesturePredictor(config["model_path"], config["class_mapping"])
    controller = GestureController(
        command_keys=config["command_keys"],
        confidence_threshold=config["confidence_threshold"],
        cooldown_seconds=config["cooldown_seconds"],
        start_enabled=config["start_enabled"],
        dry_run=config["dry_run"],
        # Hand must rest this long after a command before the next one.
        # Not in inference.yaml yet? The default is 0.5 seconds.
        rearm_seconds=config.get("rearm_seconds", 0.5),
    )
    buffer = FrameBuffer(config["window_seconds"], config["num_frames"])
    webcam = Webcam(config["camera_index"])

    log = SessionLog(
        config.get("log_dir", "data/webcam_tests/logs"),
        session_name,
        {**{key: config.get(key) for key in LOGGED_SETTINGS},
         "rearm_seconds": config.get("rearm_seconds", 0.5)},
        start_time=time.monotonic(),
    )
    print(f"Logging to {log.path}")

    last_predict_time = 0.0
    status = ["Filling buffer..."]

    try:
        while True:
            frame, now = webcam.read()

            model_frame = cv2.flip(frame, 1) if config["mirror_model_input"] else frame
            buffer.add(preprocess_frame(model_frame, image_size), now)

            if buffer.is_ready() and now - last_predict_time >= config["predict_every_seconds"]:
                last_predict_time = now
                start = time.perf_counter()
                label, command, confidence = predictor.predict(buffer.get_sequence())
                latency_ms = (time.perf_counter() - start) * 1000
                acted, reason = controller.handle(command, confidence, now)
                log.write(now, "model", label, command, confidence,
                          controller.enabled, acted, reason, latency_ms)
                if acted:
                    buffer.clear()  # start fresh so one gesture is not counted twice
                status = [
                    f"{label}  {confidence:.0%}",
                    f"Action: {command if acted else reason}",
                    f"Model time: {latency_ms:.0f} ms",
                ]

            preview = cv2.flip(frame, 1) if config["mirror_preview"] else frame
            draw_overlay(preview, controller, status)
            cv2.imshow("Gesture Controller", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("e"):
                controller.toggle()
            if ord("1") <= key <= ord("7"):
                index = key - ord("1")
                if index < len(predictor.classes):
                    item = predictor.classes[index]
                    acted, reason = controller.handle(item["command"], 1.0, now,
                                                      simulated=True)
                    log.write(now, "simulated", item["label"], item["command"],
                              1.0, controller.enabled, acted, reason)
                    status = [f"[simulated] {item['label']}",
                              f"Action: {item['command'] if acted else reason}"]
    finally:
        log.close()
        webcam.release()
        cv2.destroyAllWindows()
        print(f"Log saved: {log.path}")
        print(f"Summary:   python scripts/summarize_log.py \"{log.path}\"")
