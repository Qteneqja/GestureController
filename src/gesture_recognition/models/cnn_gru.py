"""
CNN-GRU model for classifying a sequence of gesture frames.

Input:  (batch, frames, height, width, 3), RGB values from 0 to 1
Output: (batch, num_classes), one probability per gesture
"""

import keras 
from keras import layers
from src.gesture_recognition.models.cnn import build_pretrained_frame_encoder


def build_cnn_gru(
        num_classes = 7, # Number of gesture classes
        num_frames = 16, # Number of frames in the input sequence
        image_size = (112, 112), # Size of each frame in the input sequence
        gru_units = 64, # Number of units in the GRU layer
        dense_units = 128, # Number of units in the dense layer
        dropout = 0.3, # Dropout rate
        pretrained = True # Whether to use a pretrained CNN for frame encoding
):
    """Build a CNN-GRU model that makes one prediction per video.

    The CNN extracts features from each frame. The GRU reads those
    features in sequence, allowing it to learn patterns of motion.

    Set pretrained=False for offline tests; real training should use
    pretrained ImageNet weights for comparison with CNN baseline EXP-004.
    """
    # Extract features from each frame using the CNN
    height, width = image_size
    inputs = keras.Input(
        shape=(num_frames, height, width, 3),
        name="clip"
    )

    # Encode each frame using the pretrained CNN
    frame_encoder = build_pretrained_frame_encoder(
        image_size=image_size,
        pretrained=pretrained
    )

    # Apply the same CNN to every frame, preserving the frame order.
    x = layers.TimeDistributed(frame_encoder, name="encoded_frames")(inputs)

    # read the sequence of encoded frame, preserving the frame order.
    x = layers.GRU(gru_units, name="gru")(x)
    x = layers.Dense(dense_units, activation="relu", name="dense")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(
        num_classes, activation="softmax", name="probs"
    )(x)

    return keras.Model(inputs, outputs, name="cnn_gru")
    