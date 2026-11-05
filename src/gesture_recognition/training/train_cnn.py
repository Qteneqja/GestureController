"""Train the CNN baseline.

Run from the project root:
    python src/gesture_recognition/training/train_cnn.py          # real data
    python src/gesture_recognition/training/train_cnn.py --fake   # quick test, no frames needed

All settings come from configs/cnn.yaml.

What one run produces:
    checkpoints/cnn_baseline_<id>.keras          best model 
    checkpoints/cnn_baseline_<id>_last.keras     model after the latest epoch
    experiments/cnn_baseline/<id>/history.csv    loss and accuracy per epoch
    experiments/cnn_baseline/<id>/curves.png     loss and accuracy graphs
    experiments/cnn_baseline/<id>/run_info.json  settings, versions and best score

A --fake run writes to checkpoints/fake_run/ instead, so test output never
ends up in Git.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import keras  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from src.gesture_recognition.models.cnn import (  # noqa: E402
    build_cnn_baseline,
    compile_model,
    load_num_classes,
)


def load_config(config_path):
    with open(config_path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def make_fake_datasets(num_classes, num_frames, image_size, batch_size):
    """Small random datasets with the real shapes, for testing the script."""
    import tensorflow as tf

    height, width = image_size
    rng = np.random.default_rng(0)

    def fake(count):
        clips = rng.random((count, num_frames, height, width, 3), dtype=np.float32)
        labels = rng.integers(0, num_classes, size=count).astype("int32")
        return tf.data.Dataset.from_tensor_slices((clips, labels)).batch(batch_size)

    return fake(32), fake(16)


def count_rows(csv_path):
    """Number of videos listed in a split CSV (not counting the header)."""
    with open(csv_path, encoding="utf-8") as file:
        return sum(1 for _ in file) - 1


def make_real_datasets(data_config_path, train_limit, validation_limit):
    """Load Sandeep's shared datasets, optionally using only the first videos."""
    import tensorflow as tf
    from src.gesture_recognition.data.dataset import dataset_from_config

    data_cfg = load_config(data_config_path)
    batch_size = data_cfg["batch_size"]

    datasets = []
    for split, limit in [("train", train_limit), ("validation", validation_limit)]:
        dataset = dataset_from_config(split, config_path=data_config_path)
        videos = count_rows(data_cfg[f"{split}_csv"])
        if limit:
            videos = min(videos, limit)

        # The dataset is already batched, so the video count becomes a batch
        # count. Telling Keras the count lets it show time left per epoch.
        batches = math.ceil(videos / batch_size)
        dataset = dataset.take(batches)
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(batches))

        print(f"{split}: {videos} videos in {batches} batches")
        datasets.append(dataset)

    return datasets[0], datasets[1]


def save_curves(history, path):
    """Save loss and accuracy graphs for the report."""
    import matplotlib
    matplotlib.use("Agg")  # save to file without opening a window
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["loss"]) + 1)
    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(10, 4))

    loss_ax.plot(epochs, history["loss"], label="training")
    loss_ax.plot(epochs, history["val_loss"], label="validation")
    loss_ax.set(title="Loss", xlabel="Epoch")
    loss_ax.legend()

    acc_ax.plot(epochs, history["accuracy"], label="training")
    acc_ax.plot(epochs, history["val_accuracy"], label="validation")
    acc_ax.set(title="Accuracy", xlabel="Epoch", ylim=(0, 1))
    acc_ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train_cnn(config_path="configs/cnn.yaml", fake=False):
    """Train the CNN baseline using the settings file.

    Returns (model, history) so other scripts or notebooks can call it.
    """
    config = load_config(config_path)
    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = load_config(config["data_config"])

    keras.utils.set_random_seed(train_cfg["seed"])

    num_classes = load_num_classes(config["class_mapping"])
    num_frames = data_cfg["num_frames"]
    image_size = tuple(data_cfg["image_size"])
    run_id = config["experiment_id"]

    if fake:
        train_ds, validation_ds = make_fake_datasets(
            num_classes, num_frames, image_size, data_cfg["batch_size"]
        )
        epochs = 2
        output_dir = Path(config["checkpoint_dir"]) / "fake_run"
        checkpoint_dir = output_dir
    else:
        train_ds, validation_ds = make_real_datasets(
            config["data_config"],
            train_cfg["train_limit"],
            train_cfg["validation_limit"],
        )
        epochs = train_cfg["epochs"]
        output_dir = Path(config["output_dir"]) / run_id
        checkpoint_dir = Path(config["checkpoint_dir"])

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / f"cnn_baseline_{run_id}.keras"
    last_path = checkpoint_dir / f"cnn_baseline_{run_id}_last.keras"

    model = compile_model(
        build_cnn_baseline(
            num_classes=num_classes,
            num_frames=num_frames,
            image_size=image_size,
            filters=tuple(model_cfg["filters"]),
            dense_units=model_cfg["dense_units"],
            dropout=model_cfg["dropout"],
        ),
        learning_rate=train_cfg["learning_rate"],
    )

    callbacks = [
        # Keep the model with the lowest validation loss.
        keras.callbacks.ModelCheckpoint(best_path, monitor="val_loss",
                                        save_best_only=True),
        # Also keep the latest epoch, in case a long run is interrupted.
        keras.callbacks.ModelCheckpoint(last_path),
        # Stop when validation loss stops improving, and go back to the best epoch.
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=train_cfg["early_stopping_patience"],
            restore_best_weights=True,
        ),
        # Write each epoch's numbers as soon as it finishes.
        keras.callbacks.CSVLogger(output_dir / "history.csv"),
    ]

    print(f"Training {run_id} for up to {epochs} epochs "
          f"({'fake data' if fake else 'real data'})")
    start = time.time()
    history = model.fit(train_ds, validation_data=validation_ds,
                        epochs=epochs, callbacks=callbacks,
                        shuffle=False)  # the data loader already shuffles
    minutes = (time.time() - start) / 60

    results = history.history
    best_epoch = int(np.argmin(results["val_loss"]))
    save_curves(results, output_dir / "curves.png")

    import tensorflow as tf
    run_info = {
        "experiment_id": run_id,
        "fake_data": fake,
        "epochs_run": len(results["loss"]),
        "best_epoch": best_epoch + 1,
        "best_val_loss": round(float(results["val_loss"][best_epoch]), 4),
        "best_val_accuracy": round(float(results["val_accuracy"][best_epoch]), 4),
        "training_minutes": round(minutes, 1),
        "parameters": model.count_params(),
        "best_model": str(best_path),
        "tensorflow_version": tf.__version__,
        "keras_version": keras.__version__,
        "settings": config,
    }
    with open(output_dir / "run_info.json", "w", encoding="utf-8") as file:
        json.dump(run_info, file, indent=2)

    print(f"Done in {minutes:.1f} minutes. "
          f"Best validation accuracy: {run_info['best_val_accuracy']:.3f} "
          f"(epoch {run_info['best_epoch']})")
    print(f"Best model saved to {best_path}")
    print(f"Results saved to {output_dir}")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the CNN baseline.")
    parser.add_argument("--config", default="configs/cnn.yaml",
                        help="settings file (default: configs/cnn.yaml)")
    parser.add_argument("--fake", action="store_true",
                        help="use random data to test the script without frames")
    args = parser.parse_args()

    # Shared paths in the config files are relative to the project root.
    os.chdir(PROJECT_ROOT)
    train_cnn(args.config, fake=args.fake)