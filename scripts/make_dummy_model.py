"""Build a placeholder model so the webcam app can be tested early.

It has the same input and output shape as the real models:
  input  (16, 112, 112, 3)
  output 7 class probabilities
It has NOT been trained, so its predictions are meaningless. Its
confidence stays around 14% (1 in 7), so it should never pass the
confidence threshold. That is a useful safety check by itself.

Usage:
    python scripts/make_dummy_model.py
"""

from pathlib import Path

import keras

project_root = Path(__file__).resolve().parent.parent
output_path = project_root / "checkpoints" / "dummy_model.keras"
output_path.parent.mkdir(exist_ok=True)

model = keras.Sequential([
    keras.Input(shape=(16, 112, 112, 3)),
    keras.layers.GlobalAveragePooling3D(),
    keras.layers.Dense(7, activation="softmax"),
])
model.save(output_path)
print(f"Saved placeholder model to {output_path}")
