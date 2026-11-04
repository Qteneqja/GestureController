import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

steps = [
    "filter_labels.py",
    "create_splits.py",
    "prepare_test_labels.py",
    "split_statistics.py",
]

for script in steps:
    print(f"\nRunning {script}...", flush=True)

    # Stop if a preparation step fails.
    subprocess.run(
        [sys.executable, str(project_root / "scripts" / script)],
        cwd=project_root,
        check=True,
    )

print("\nDataset split preparation completed.")