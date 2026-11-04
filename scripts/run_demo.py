"""Start the webcam gesture controller.

Usage (from the project folder, with .venv active):
    python scripts/run_demo.py
    python scripts/run_demo.py --config configs/inference.yaml
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from gesture_recognition.application.main import run  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=str(project_root / "configs" / "inference.yaml"))
args = parser.parse_args()

# Relative paths in the config are read from the project folder.
import os  # noqa: E402

os.chdir(project_root)
run(args.config)
