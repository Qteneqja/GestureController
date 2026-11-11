"""Find where the CNN baseline makes mistakes.

Runs a trained model on the Validation set and saves:
    per_class.csv         accuracy for each gesture and what it is confused with
    confusion_matrix.csv  counts: rows = true gesture, columns = predicted gesture
    confusion_matrix.png  the same matrix as percentages per row
    summary.json          overall accuracy and the main error types

The test set is not used here. It is kept for the final evaluation.

Run from the project root:
    python src/gesture_recognition/training/analyze_cnn.py --model checkpoints/cnn_baseline_EXP-004.keras --output experiments/cnn_baseline/EXP-004/weaknesses
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

# Swipe gestures by class ID (see config/gesture_classes.json).
# Direction error: right hand shape, wrong direction.
DIRECTION_PAIRS = [(0, 1), (2, 3)]   # Swiping L/R, Two Fingers L/R
# Hand-shape error: right direction, wrong hand shape.
HAND_SHAPE_PAIRS = [(0, 2), (1, 3)]  # Left: swipe vs two fingers, Right: same


def load_class_names(class_mapping_path):
    with open(class_mapping_path, encoding="utf-8") as file:
        classes = json.load(file)["classes"]
    return [c["label"] for c in sorted(classes, key=lambda c: c["id"])]


def confusion_counts(y_true, y_pred, num_classes):
    """Rows = true gesture, columns = predicted gesture."""
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[true, pred] += 1
    return matrix


def summarize_errors(matrix, class_names):
    """Turn a confusion matrix into per-gesture results and error types."""
    totals = matrix.sum(axis=1)
    correct = np.diag(matrix)
    errors_total = int(matrix.sum() - correct.sum())

    per_class = []
    for i, name in enumerate(class_names):
        row = matrix[i].copy()
        row[i] = 0
        worst = int(row.argmax())
        per_class.append({
            "gesture": name,
            "videos": int(totals[i]),
            "correct": int(correct[i]),
            "accuracy": round(float(correct[i] / totals[i]), 4) if totals[i] else 0.0,
            "most_confused_with": class_names[worst] if row[worst] else "",
            "confused_share": round(float(row[worst] / totals[i]), 4) if totals[i] else 0.0,
        })

    def pair_errors(pairs):
        return int(sum(matrix[a, b] + matrix[b, a] for a, b in pairs))

    # Biggest single mix-ups, e.g. "Swiping Left -> Swiping Right".
    mixups = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and matrix[i, j]:
                mixups.append({
                    "true": class_names[i],
                    "predicted": class_names[j],
                    "videos": int(matrix[i, j]),
                })
    mixups.sort(key=lambda m: m["videos"], reverse=True)

    direction = pair_errors(DIRECTION_PAIRS)
    hand_shape = pair_errors(HAND_SHAPE_PAIRS)
    return {
        "videos": int(matrix.sum()),
        "accuracy": round(float(correct.sum() / matrix.sum()), 4),
        "total_errors": errors_total,
        "direction_errors": direction,
        "hand_shape_errors": hand_shape,
        "direction_share_of_errors": round(direction / errors_total, 4) if errors_total else 0.0,
        "hand_shape_share_of_errors": round(hand_shape / errors_total, 4) if errors_total else 0.0,
        "top_mixups": mixups[:5],
        "per_class": per_class,
    }


def save_matrix_png(matrix, class_names, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    totals = matrix.sum(axis=1, keepdims=True)
    percent = np.divide(matrix, totals, out=np.zeros(matrix.shape),
                        where=totals > 0) * 100

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(percent, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(class_names)), class_names, rotation=40, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted gesture")
    ax.set_ylabel("True gesture")
    ax.set_title(title)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{percent[i, j]:.0f}%", ha="center", va="center",
                    color="white" if percent[i, j] > 50 else "black", fontsize=9)
    fig.colorbar(image, ax=ax, label="% of true gesture")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def predict_split(model, dataset):
    """Return true and predicted class IDs for every video in the dataset."""
    y_true, y_pred = [], []
    for batch_number, (clips, labels) in enumerate(dataset, start=1):
        probabilities = model.predict_on_batch(clips)
        y_true.extend(np.asarray(labels).tolist())
        y_pred.extend(np.argmax(probabilities, axis=1).tolist())
        if batch_number % 100 == 0:
            print(f"  {len(y_true)} videos checked")
    return y_true, y_pred


def analyze_cnn(model_path, output_dir, data_config="configs/data.yaml",
                limit=None):
    """Run the analysis and save the results. Returns the summary."""
    import keras
    import yaml
    from src.gesture_recognition.data.dataset import dataset_from_config

    with open(data_config, encoding="utf-8") as file:
        data_cfg = yaml.safe_load(file)
    class_names = load_class_names(data_cfg["class_mapping"])

    model = keras.models.load_model(model_path)
    dataset = dataset_from_config("validation", config_path=data_config)
    if limit:
        dataset = dataset.take(int(np.ceil(limit / data_cfg["batch_size"])))

    print(f"Checking {model_path} on the validation set...")
    y_true, y_pred = predict_split(model, dataset)

    matrix = confusion_counts(y_true, y_pred, len(class_names))
    summary = summarize_errors(matrix, class_names)
    summary["model"] = str(model_path)
    summary["split"] = "validation"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "per_class.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary["per_class"][0]))
        writer.writeheader()
        writer.writerows(summary["per_class"])

    with open(output_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true \\ predicted"] + class_names)
        for name, row in zip(class_names, matrix):
            writer.writerow([name] + row.tolist())

    save_matrix_png(matrix, class_names, output_dir / "confusion_matrix.png",
                    f"CNN baseline, validation set ({summary['accuracy']:.1%} accuracy)")

    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"\nAccuracy: {summary['accuracy']:.1%} on {summary['videos']} videos")
    print("\nPer gesture:")
    for row in summary["per_class"]:
        confused = (f"  (most confused with {row['most_confused_with']}, "
                    f"{row['confused_share']:.0%})" if row["most_confused_with"] else "")
        print(f"  {row['gesture']:<27} {row['accuracy']:>6.1%}{confused}")
    print(f"\nOf {summary['total_errors']} mistakes:")
    print(f"  wrong direction (left vs right): {summary['direction_errors']} "
          f"({summary['direction_share_of_errors']:.0%})")
    print(f"  wrong hand shape (swipe vs two fingers): {summary['hand_shape_errors']} "
          f"({summary['hand_shape_share_of_errors']:.0%})")
    print(f"\nResults saved to {output_dir}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find where the CNN baseline makes mistakes.")
    parser.add_argument("--model", required=True, help="path to a trained .keras file")
    parser.add_argument("--output", required=True, help="folder for the results")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--limit", type=int, default=None,
                        help="only check this many videos (for a quick test)")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    analyze_cnn(args.model, args.output, args.data_config, args.limit)