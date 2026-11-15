"""Start the webcam gesture controller.

Usage (from the project folder, with .venv active):
    python scripts/run_demo.py
    python scripts/run_demo.py --session swipe_right_x5
    python scripts/run_demo.py --config configs/inference.yaml
    python scripts/run_demo.py --window 3.0 --session w30_swipe_right_x5

--session names the log file, so you can tell your tests apart later.
"""

import argparse
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from gesture_recognition.application.main import run  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=str(project_root / "configs" / "inference.yaml"))
parser.add_argument("--session", default="test",
                    help="short name for this test, used in the log file name")
parser.add_argument("--window", type=float, default=None,
                    help="window length in seconds (overrides the config file)")
args = parser.parse_args()

# Relative paths in the config are read from the project folder.
os.chdir(project_root)
run(args.config, args.session, args.window)
