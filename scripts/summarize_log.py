"""Summarize one webcam test log.

Usage:
    python scripts/summarize_log.py data/webcam_tests/logs/<file>.csv

Prints:
  - how many commands fired, and which ones
  - how often each gesture was the model's top prediction, and its best %
  - commands per minute (use a "talking only" session for false triggers)
  - model time: median, average and 95th percentile (slowest 5%),
    skipping the first prediction (warm-up)

Only real model predictions are counted. Number-key simulations are
listed separately so they never mix into the results.
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

if len(sys.argv) != 2:
    sys.exit("Usage: python scripts/summarize_log.py <log.csv>")

path = Path(sys.argv[1])
with open(path, encoding="utf-8", newline="") as file:
    rows = list(csv.DictReader(file))

model_rows = [r for r in rows if r["source"] == "model"]
simulated = [r for r in rows if r["source"] == "simulated"]
if not model_rows:
    sys.exit("No model predictions in this log yet.")

info_path = path.with_suffix(".json")
if info_path.exists():
    info = json.loads(info_path.read_text(encoding="utf-8"))
    print(f"Session: {info.get('session')}   started {info.get('started')}")
    for key in ("model_path", "window_seconds", "confidence_threshold",
                "cooldown_seconds", "mirror_model_input"):
        print(f"  {key}: {info.get(key)}")

times = [float(r["time_s"]) for r in model_rows]
minutes = max((times[-1] - times[0]) / 60, 1e-9)
fired = [r for r in model_rows if r["fired"] == "1"]

print(f"\nLength: {minutes:.1f} min, {len(model_rows)} predictions")

print(f"\nCommands fired: {len(fired)}  ({len(fired) / minutes:.1f} per minute)")
for command, count in Counter(r["command"] for r in fired).most_common():
    print(f"  {command}: {count}")

print("\nTop prediction per gesture (count, best confidence):")
counts = Counter(r["label"] for r in model_rows)
best = defaultdict(float)
for r in model_rows:
    best[r["label"]] = max(best[r["label"]], float(r["confidence"]))
for label, count in counts.most_common():
    print(f"  {label:28s} {count:5d}   best {best[label]:.0%}")

print("\nWhy commands did not fire:")
for reason, count in Counter(r["reason"] for r in model_rows
                             if r["fired"] != "1").most_common():
    print(f"  {reason}: {count}")

ms = np.array([float(r["model_ms"]) for r in model_rows if r["model_ms"]])
if ms.size > 1:
    # The first prediction is always slow because TensorFlow sets itself up
    # on the first call. Skip it so it does not distort the numbers.
    warmup, ms = ms[0], ms[1:]
    print(f"\nModel time: median {np.median(ms):.0f} ms, "
          f"average {ms.mean():.0f} ms, "
          f"95th percentile {np.percentile(ms, 95):.0f} ms")
    print(f"  (first prediction skipped as warm-up: {warmup:.0f} ms)")

if simulated:
    print(f"\n(Ignored {len(simulated)} simulated number-key presses.)")
