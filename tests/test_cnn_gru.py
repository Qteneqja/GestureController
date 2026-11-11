"""Tests for the CNN-GRU architecture. No dataset download is needed."""

import numpy as np
import keras

from src.gesture_recognition.models.cnn_gru import build_cnn_gru

# Test functions for the CNN-GRU model
def test_input_and_output_shapes():
    model = build_cnn_gru(pretrained=False)

    assert model.input_shape == (None, 16, 112, 112, 3)
    assert model.output_shape == (None, 7)

# Test that the model outputs valid probability distributions
def test_predictions_are_probabilities():
    model = build_cnn_gru(pretrained=False)
    clips = np.random.default_rng(42).random(
        (1, 16, 112, 112, 3), dtype=np.float32
    )

    probabilities = np.asarray(model(clips, training=False))

    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)

# Test that the saved model can be loaded and produces the same predictions
def test_saved_model_can_be_loaded(tmp_path):
    model = build_cnn_gru(pretrained=False)
    path = tmp_path / "cnn_gru.keras"
    model.save(path)

    reloaded = keras.models.load_model(path)
    assert isinstance(reloaded, keras.Model)
    clips = np.random.default_rng(42).random(
        (1, 16, 112, 112, 3), dtype=np.float32
    )

    original = np.asarray(model(clips, training=False))
    restored = np.asarray(reloaded(clips, training=False))
    assert np.allclose(original, restored, atol=1e-5)
