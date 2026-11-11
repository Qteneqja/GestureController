"""Train the CNN-GRU model.

Run from the project root:

    python src/gesture_recognition/training/train_cnn_gru.py --fake
    python src/gesture_recognition/training/train_cnn_gru.py

Use --fake first to test the training pipeline without the Jester dataset.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import keras
import numpy as np
import tensorflow as tf
import yaml

from src.gesture_recognition.models.cnn import compile_model, load_num_classes
from src.gesture_recognition.models.cnn_gru import build_cnn_gru
from src.gesture_recognition.training.train_cnn import (
    make_fake_datasets,
    make_real_datasets,
    save_curves,
)

def load_config(path):
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)

def train_cnn_gru(config_path="configs/cnn_gru.yaml", fake=False):
    """Train the model and save its checkpoint and experiment records."""
    config = load_config(config_path)
    data_config = load_config(config["data_config"])
    model_config = config["model"]
    training_config = config["training"]

    keras.utils.set_random_seed(training_config["seed"])

    num_classes = load_num_classes(config["class_mapping"])
    num_frames = data_config["num_frames"]
    image_size = tuple(data_config["image_size"])
    run_id = config["experiment_id"]

    if fake:
        train_data, validation_data = make_fake_datasets(
            num_classes,
            num_frames,
            image_size,
            data_config["batch_size"],
            )
        epochs = 2
        output_dir = Path(config["checkpoint_dir"]) / "fake_cnn_gru"
        checkpoint_dir = output_dir
    else:
        train_data, validation_data = make_real_datasets(
            config["data_config"],
            training_config["train_limit"],
            training_config["validation_limit"],
        )
        epochs = training_config["epochs"]
        output_dir = Path(config["output_dir"]) / run_id
        checkpoint_dir = Path(config["checkpoint_dir"])

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_path = checkpoint_dir / f"cnn_gru_{run_id}.keras"
    last_path = checkpoint_dir / f"cnn_gru_{run_id}_last.keras"

    model = build_cnn_gru(
        num_classes=num_classes,
        num_frames=num_frames,
        image_size=image_size,
        gru_units=model_config["gru_units"],
        dense_units=model_config["dense_units"],
        dropout=model_config["dropout"],
        # Fake runs do not need an ImageNet weight download.
        pretrained=model_config["pretrained"] and not fake,
    )
    compile_model(model, learning_rate=training_config["learning_rate"])

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            best_path, monitor="val_loss", save_best_only=True
        ),
        keras.callbacks.ModelCheckpoint(last_path),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=training_config["early_stopping_patience"],
            restore_best_weights=True,
        ),
        keras.callbacks.CSVLogger(output_dir / "history.csv"),
    ]

    print(
        f"Training {run_id} for up to {epochs} epochs "
        f"({'fake data' if fake else 'real data'})"
    )
    start = time.time()

    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        callbacks=callbacks,
        shuffle=False,  # The shared data loader handles training shuffle.
    )

    training_minutes = (time.time() - start) / 60
    results = history.history
    best_epoch = int(np.argmin(results["val_loss"]))
    save_curves(results, output_dir / "curves.png")

    run_info = {
        "experiment_id": run_id,
        "fake_data": fake,
        "epochs_run": len(results["loss"]),
        "best_epoch": best_epoch + 1,
        "best_val_loss": float(results["val_loss"][best_epoch]),
        "best_val_accuracy": float(results["val_accuracy"][best_epoch]),
        "training_minutes": training_minutes,
        "parameters": model.count_params(),
        "best_model": str(best_path),
        "tensorflow_version": tf.__version__,
        "keras_version": keras.__version__,
        "settings": config,
        "data_settings": data_config,
    }

    with open(output_dir / "run_info.json", "w", encoding="utf-8") as file:
        json.dump(run_info, file, indent=2)

    print(
        f"Done in {training_minutes:.1f} minutes. "
        f"Best validation accuracy: {run_info['best_val_accuracy']:.3f} "
        f"(epoch {run_info['best_epoch']})"
    )
    print(f"Best model saved to {best_path}")
    print(f"Results saved to {output_dir}")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the CNN-GRU model.")
    parser.add_argument(
        "--config",
        default="configs/cnn_gru.yaml",
        help="settings file (default: configs/cnn_gru.yaml)",
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="test the pipeline with random data instead of video frames",
    )
    args = parser.parse_args()

    # Configuration paths are relative to the project root.
    os.chdir(PROJECT_ROOT)
    train_cnn_gru(args.config, fake=args.fake)