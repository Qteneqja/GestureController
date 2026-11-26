"""Combine interrupted CNN-GRU histories without changing their source files.

Run from any directory with Python and Matplotlib:
    python experiments/cnn_gru_baseline/combine_histories.py

CSV epoch numbers remain zero-based, matching Keras CSVLogger. Charts show
human-readable one-based epochs, with dashed lines at restart boundaries.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FIELDS = ("epoch", "accuracy", "loss", "val_accuracy", "val_loss")
RUNS = {
    "EXP-002": (
        ("EXP-002", "EXP-002/history.csv"),
        ("EXP-002-continuation-01", "EXP-002-continuation-01/history.csv"),
    ),
    "EXP-003": (
        ("EXP-003-retry-01", "EXP-003-retry-01/history.csv"),
        ("EXP-003-retry-01-continuation-01", "EXP-003-retry-01-continuation-01/history.csv"),
        ("EXP-003-retry-01-continuation-02-local", "EXP-003-retry-01-continuation-02-local/history.csv"),
    ),
}


def combine(experiment, segments):
    rows = []
    restart_after = []
    for segment, relative_path in segments:
        if rows:
            restart_after.append(len(rows))
        with (ROOT / relative_path).open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError(f"Unexpected columns in {relative_path}")
            for row in reader:
                if int(row["epoch"]) != len(rows):
                    raise ValueError(f"Missing or duplicate epoch in {relative_path}")
                for field in FIELDS[1:]:
                    float(row[field])
                rows.append({**row, "source_segment": segment})

    output = ROOT / experiment
    output.mkdir(exist_ok=True)
    with (output / f"{experiment}_history.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=(*FIELDS, "source_segment"))
        writer.writeheader()
        writer.writerows(rows)

    epochs = [int(row["epoch"]) + 1 for row in rows]
    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, train, valid, title in (
        (loss_ax, "loss", "val_loss", "Loss"),
        (acc_ax, "accuracy", "val_accuracy", "Accuracy"),
    ):
        ax.plot(epochs, [float(row[train]) for row in rows], label="Training")
        ax.plot(epochs, [float(row[valid]) for row in rows], label="Validation")
        for index, boundary in enumerate(restart_after):
            ax.axvline(boundary + 0.5, color="gray", linestyle="--", linewidth=1,
                       label="Restart" if index == 0 else None)
        ax.set(title=title, xlabel="Epoch")
        ax.legend()
    acc_ax.set_ylim(0, 1)
    fig.suptitle(f"{experiment}: combined training history")
    fig.tight_layout()
    fig.savefig(output / f"{experiment}_curves.png", dpi=120)
    plt.close(fig)

    best = min(rows, key=lambda row: float(row["val_loss"]))
    print(f"{experiment}: {len(rows)} epochs; best loss at epoch {int(best['epoch']) + 1}; "
          f"restart boundaries after {restart_after}")


if __name__ == "__main__":
    for name, parts in RUNS.items():
        combine(name, parts)
