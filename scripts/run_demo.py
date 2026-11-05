"""Start the webcam gesture controller.

Usage (from the project folder, with .venv active):
    python scripts/run_demo.py
    python scripts/run_demo.py --session swipe_right_x5
    python scripts/run_demo.py --config configs/inference.yaml

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
args = parser.parse_args()

# Relative paths in the config are read from the project folder.
os.chdir(project_root)
run(args.config, args.session)
