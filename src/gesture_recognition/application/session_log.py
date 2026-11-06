"""Save every prediction to a CSV file so live tests can be counted later.

One file per run, in data/webcam_tests/logs/ (that folder is ignored by
Git, so logs stay on your computer). A small JSON file next to it records
the settings used, so you know which test each log belongs to.

Columns:
  time_s       seconds since the app started
  source       "model" (real prediction) or "simulated" (number key)
  label        predicted gesture, e.g. "Swiping Right"
  command      the command linked to that gesture, e.g. "next_slide"
  confidence   0 to 1
  enabled      1 if commands were switched on
  fired        1 if the command was actually sent (or printed in dry run)
  reason       why it did or did not fire: ok, low confidence, cooldown, ...
  model_ms     how long the model took for this prediction
"""

import csv
import json
from datetime import datetime
from pathlib import Path

COLUMNS = ["time_s", "source", "label", "command", "confidence",
           "enabled", "fired", "reason", "model_ms"]


class SessionLog:
    def __init__(self, log_dir, session_name, settings, start_time):
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_"
                            for c in session_name)
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        self.path = log_dir / f"{stamp}_{safe_name}.csv"
        self.start_time = start_time
        self.file = open(self.path, "w", encoding="utf-8", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(COLUMNS)

        info = {"session": session_name, "started": stamp, **settings}
        with open(self.path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

    def write(self, now, source, label, command, confidence, enabled,
              fired, reason, model_ms=None):
        self.writer.writerow([
            round(now - self.start_time, 3), source, label, command,
            round(confidence, 4), int(enabled), int(fired), reason,
            "" if model_ms is None else round(model_ms, 1),
        ])
        self.file.flush()  # save straight away, in case the app crashes

    def close(self):
        self.file.close()
