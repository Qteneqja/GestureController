"""CNN baseline model.

How it works:
1. A CNN (the "frame encoder") looks at ONE frame and turns it into a list
   of numbers that describe what it sees (hand shape, position).
2. The same encoder is applied to all 16 frames of a clip.
3. The 16 frames are combined without using their order, then one
   prediction is made for the whole clip: 7 probabilities, one per gesture.

This model does NOT know the order of the frames. That is on purpose:
it is the baseline the CNN-GRU is compared against.

Two settings (chosen in configs/cnn.yaml):

aggregation: how the 16 frames are combined
  "features"     average what each frame sees, then make one prediction
                 (default; trains much better)
  "predictions"  each frame makes its own prediction, then average them
                 (used in EXP-001; stayed at chance level on real data)

backbone: which frame encoder to use
  "small"        small CNN trained from scratch (about 100k weights)
  "mobilenet"    MobileNetV2 pretrained on ImageNet photos; already knows
                 edges and shapes, so it usually learns faster

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
    """Build the small CNN that turns one frame into a feature vector.

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


def build_pretrained_frame_encoder(image_size=(112, 112), width=0.35,
                                   pretrained=True, name="frame_encoder"):
    """Build a frame encoder from MobileNetV2, pretrained on ImageNet photos.

    width: 0.35 is the smallest and fastest version (good for CPU).
    pretrained: True downloads the ImageNet weights the first time
                (about 2 MB, needs internet). False starts from random
                weights, which is only useful for quick tests.

    Input:  (height, width, 3) values 0 to 1
    Output: (1280,) numbers
    """
    height, width_px = image_size
    inputs = keras.Input(shape=(height, width_px, 3), name="frame")

    # MobileNetV2 expects pixel values from -1 to 1, ours are 0 to 1.
    x = layers.Rescaling(2.0, offset=-1.0, name="to_minus1_1")(inputs)

    backbone = keras.applications.MobileNetV2(
        input_shape=(height, width_px, 3),
        alpha=width,
        include_top=False,
        weights="imagenet" if pretrained else None,
        pooling="avg",
    )
    outputs = backbone(x)
    return keras.Model(inputs, outputs, name=name)


def build_cnn_baseline(num_classes=7, num_frames=16, image_size=(112, 112),
                       filters=(16, 32, 64, 128), dense_units=128,
                       dropout=0.3, aggregation="features", backbone="small",
                       pretrained=True):
    """Build the full CNN baseline for whole video clips.

    Input:  (batch, num_frames, height, width, 3)
    Output: (batch, num_classes) probabilities
    """
    if aggregation not in ("features", "predictions"):
        raise ValueError('aggregation must be "features" or "predictions"')
    if backbone not in ("small", "mobilenet"):
        raise ValueError('backbone must be "small" or "mobilenet"')

    height, width = image_size
    inputs = keras.Input(shape=(num_frames, height, width, 3), name="clip")

    if backbone == "small":
        encoder = build_frame_encoder(image_size, filters)
    else:
        encoder = build_pretrained_frame_encoder(image_size,
                                                 pretrained=pretrained)

    # Run the same frame encoder on every frame -> (batch, 16, features)
    x = layers.TimeDistributed(encoder, name="encode_frames")(inputs)

    if aggregation == "features":
        # Average what the 16 frames see -> (batch, features),
        # then make one prediction for the clip.
        x = layers.GlobalAveragePooling1D(name="average_frames")(x)
        x = layers.Dense(dense_units, activation="relu", name="dense")(x)
        x = layers.Dropout(dropout, name="dropout")(x)
        outputs = layers.Dense(num_classes, activation="softmax",
                               name="probs")(x)
    else:
        # Each frame makes its own 7-way guess, then the 16 guesses
        # are averaged into one clip prediction.
        x = layers.TimeDistributed(
            layers.Dense(dense_units, activation="relu"), name="frame_dense"
        )(x)
        x = layers.Dropout(dropout, name="dropout")(x)
        x = layers.TimeDistributed(
            layers.Dense(num_classes, activation="softmax"),
            name="frame_probs",
        )(x)
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