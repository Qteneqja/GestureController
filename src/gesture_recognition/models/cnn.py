"""CNN baseline model (Member 2).

How it works:
1. A small CNN (the "frame encoder") looks at ONE frame and turns it into
   a list of numbers that describe the hand shape in that frame.
2. The same encoder is applied to all 16 frames of a clip.
3. Each frame gets its own guess: 7 probabilities, one per gesture.
4. The 16 guesses are averaged into one prediction for the whole clip.

This model does NOT know the order of the frames. That is on purpose:
it is the baseline the CNN-GRU (Member 3) is compared against.

Input:  (batch, 16, 112, 112, 3)  float32 frames, values 0 to 1
Output: (batch, 7)                probabilities that add up to 1

Only built-in Keras layers are used, so the saved .keras file loads with
plain keras.models.load_model() in the webcam app.
"""

import json

import keras
from keras import layers


def load_num_classes(class_mapping_path="config/gesture_classes.json"):
    """Read how many gesture classes the team uses (currently 7)."""
    with open(class_mapping_path, encoding="utf-8") as file:
        return len(json.load(file)["classes"])


def build_frame_encoder(image_size=(112, 112), filters=(16, 32, 64, 128),
                        name="frame_encoder"):
    """Build the CNN that turns one frame into a feature vector.

    Each block: convolution (finds patterns) -> batch normalization
    (keeps training stable) -> ReLU -> max pooling (halves the image size).
    The last step averages each feature map into one number, so the
    output is a vector with one value per filter in the last block.

    Member 3 can reuse this function inside the CNN-GRU, so both models
    see frames the same way and the only difference is how they combine
    the 16 frames.

    Input:  (height, width, 3)
    Output: (filters[-1],)   e.g. 128 numbers
    """
    height, width = image_size
    inputs = keras.Input(shape=(height, width, 3), name="frame")
    x = inputs

    for index, count in enumerate(filters, start=1):
        x = layers.Conv2D(count, 3, padding="same", use_bias=False,
                          name=f"conv{index}")(x)
        x = layers.BatchNormalization(name=f"bn{index}")(x)
        x = layers.ReLU(name=f"relu{index}")(x)
        x = layers.MaxPooling2D(2, name=f"pool{index}")(x)

    outputs = layers.GlobalAveragePooling2D(name="features")(x)
    return keras.Model(inputs, outputs, name=name)


def build_cnn_baseline(num_classes=7, num_frames=16, image_size=(112, 112),
                       filters=(16, 32, 64, 128), dense_units=128,
                       dropout=0.3):
    """Build the full CNN baseline for whole video clips.

    Input:  (batch, num_frames, height, width, 3)
    Output: (batch, num_classes) probabilities
    """
    height, width = image_size
    inputs = keras.Input(shape=(num_frames, height, width, 3), name="clip")

    # Run the same frame encoder on every frame -> (batch, 16, features)
    encoder = build_frame_encoder(image_size, filters)
    x = layers.TimeDistributed(encoder, name="encode_frames")(inputs)

    # Turn each frame's features into its own 7-way guess
    x = layers.TimeDistributed(
        layers.Dense(dense_units, activation="relu"), name="frame_dense"
    )(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    x = layers.TimeDistributed(
        layers.Dense(num_classes, activation="softmax"), name="frame_probs"
    )(x)

    # Average the 16 frame guesses into one clip prediction -> (batch, 7)
    outputs = layers.GlobalAveragePooling1D(name="average_frames")(x)

    return keras.Model(inputs, outputs, name="cnn_baseline")


def compile_model(model, learning_rate=0.001):
    """Set the optimizer, loss and metric used for training.

    Labels are integer class IDs (0 to 6), so we use sparse categorical
    cross-entropy.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    # Quick self-check with fake data. Run from the project root:
    #   python src/gesture_recognition/models/cnn.py
    import numpy as np

    num_classes = load_num_classes()
    model = compile_model(build_cnn_baseline(num_classes=num_classes))
    model.summary()

    fake_clips = np.random.rand(2, 16, 112, 112, 3).astype("float32")
    fake_labels = np.array([0, 1])

    predictions = model.predict(fake_clips, verbose=0)
    print("Output shape:", predictions.shape)
    print("Each row adds up to:", predictions.sum(axis=1).round(3))

    model.fit(fake_clips, fake_labels, epochs=1, verbose=0)
    print("One training step on fake data worked.") 