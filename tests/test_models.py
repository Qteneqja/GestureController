"""Tests for the CNN baseline. No video frames needed.

Run from the project root:
    python -m pytest tests/test_models.py
"""

import sys
from pathlib import Path

import numpy as np
import yaml

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import keras  # noqa: E402

from src.gesture_recognition.models.cnn import (  # noqa: E402
    build_cnn_baseline,
    build_frame_encoder,
    compile_model,
    load_num_classes,
)
from src.gesture_recognition.training.train_cnn import train_cnn  # noqa: E402

CLASS_FILE = project_root / "config" / "gesture_classes.json"


def fake_clips(count=2):
    return np.random.rand(count, 16, 112, 112, 3).astype("float32")


def test_class_count_comes_from_shared_file():
    assert load_num_classes(CLASS_FILE) == 7


def test_frame_encoder_turns_one_frame_into_features():
    encoder = build_frame_encoder()
    frame = np.random.rand(1, 112, 112, 3).astype("float32")
    features = encoder.predict(frame, verbose=0)
    # One feature per filter in the last block (128 by default).
    assert features.shape == (1, 128)


def test_model_matches_agreed_input_and_output():
    model = build_cnn_baseline()
    assert model.input_shape == (None, 16, 112, 112, 3)
    assert model.output_shape == (None, 7)


def test_output_is_probabilities():
    model = build_cnn_baseline()
    probabilities = model.predict(fake_clips(), verbose=0)
    # The webcam app reads the top value as a confidence, so each row
    # must be valid probabilities that add up to 1.
    assert np.all(probabilities >= 0) and np.all(probabilities <= 1)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)


def test_settings_change_the_model():
    small = build_cnn_baseline(filters=(8, 16))
    default = build_cnn_baseline()
    assert small.count_params() < default.count_params()


def test_one_training_step_runs():
    model = compile_model(build_cnn_baseline())
    history = model.fit(fake_clips(), np.array([0, 6]), epochs=1, verbose=0)
    assert np.isfinite(history.history["loss"][0])


def test_saved_model_loads_like_the_webcam_app(tmp_path):
    model = build_cnn_baseline()
    path = tmp_path / "cnn_baseline.keras"
    model.save(path)

    # The webcam app uses plain load_model, with no custom code.
    reloaded = keras.models.load_model(path)
    clips = fake_clips()
    assert np.allclose(model.predict(clips, verbose=0),
                       reloaded.predict(clips, verbose=0), atol=1e-5)


def test_training_script_runs_end_to_end(tmp_path, monkeypatch):
    # Use the real settings file, but send all output to a temporary folder.
    monkeypatch.chdir(project_root)
    with open("configs/cnn.yaml", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config["checkpoint_dir"] = str(tmp_path)
    config_path = tmp_path / "cnn_test.yaml"
    with open(config_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file)

    model, history = train_cnn(config_path, fake=True)

    output = tmp_path / "fake_run"
    run_id = config["experiment_id"]
    assert (output / f"cnn_baseline_{run_id}.keras").exists()
    assert (output / "history.csv").exists()
    assert (output / "curves.png").exists()
    assert (output / "run_info.json").exists()
    assert len(history.history["loss"]) == 2