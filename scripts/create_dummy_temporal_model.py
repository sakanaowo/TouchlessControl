#!/usr/bin/env python3
"""Create an untrained GRU-64 TFLite model for development and testing.

This script builds the GRU-64 architecture specified in the design doc,
with random (untrained) weights, and converts it to TFLite format.

Output: model/temporal_classifier/temporal_classifier.tflite

Usage:
    conda activate sign
    python scripts/create_dummy_temporal_model.py
"""

from __future__ import annotations

import os

import numpy as np
import tensorflow as tf

WINDOW_SIZE = 20
NUM_FEATURES = 93
NUM_CLASSES = 5
HIDDEN_UNITS = 64

OUTPUT_DIR = "model/temporal_classifier"
TFLITE_PATH = os.path.join(OUTPUT_DIR, "temporal_classifier.tflite")
KERAS_PATH = os.path.join(OUTPUT_DIR, "temporal_classifier.keras")


def build_gru64_model() -> tf.keras.Model:
    """Build the GRU-64 sequential model (untrained)."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((WINDOW_SIZE, NUM_FEATURES)),
            tf.keras.layers.GRU(
                HIDDEN_UNITS,
                unroll=True,
                return_sequences=False,
            ),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def convert_to_tflite(model: tf.keras.Model) -> bytes:
    """Convert a Keras model to dynamically-quantized TFLite."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return converter.convert()


def verify_tflite(tflite_path: str) -> None:
    """Load the TFLite model and run one dummy inference to verify shapes."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = tuple(input_details[0]["shape"])
    output_shape = tuple(output_details[0]["shape"])

    print(f"  Input shape:  {input_shape}")
    print(f"  Output shape: {output_shape}")
    assert input_shape == (
        1,
        WINDOW_SIZE,
        NUM_FEATURES,
    ), f"Unexpected input: {input_shape}"
    assert output_shape == (1, NUM_CLASSES), f"Unexpected output: {output_shape}"

    # Run dummy inference
    dummy_input = np.random.randn(1, WINDOW_SIZE, NUM_FEATURES).astype(np.float32)
    interpreter.set_tensor(input_details[0]["index"], dummy_input)
    interpreter.invoke()
    scores = np.squeeze(interpreter.get_tensor(output_details[0]["index"]))
    print(f"  Dummy scores: {scores}")
    print(f"  Sum ≈ 1.0:    {scores.sum():.4f}")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Building GRU-64 model...")
    model = build_gru64_model()
    model.summary()

    print(f"\nSaving Keras model → {KERAS_PATH}")
    model.save(KERAS_PATH)

    print(f"Converting to TFLite → {TFLITE_PATH}")
    tflite_bytes = convert_to_tflite(model)
    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_bytes)
    size_kb = os.path.getsize(TFLITE_PATH) / 1024
    print(f"  TFLite size: {size_kb:.1f} KB")

    print("\nVerifying TFLite model:")
    verify_tflite(TFLITE_PATH)

    print("\nDone! Model ready for development.")


if __name__ == "__main__":
    main()
