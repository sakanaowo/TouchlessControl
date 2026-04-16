"""Tests for the dummy GRU-64 TFLite model and SequenceBuffer → TemporalClassifier integration."""

from __future__ import annotations

import os

import numpy as np
import pytest

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf

    Interpreter = tf.lite.Interpreter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TFLITE_PATH = "model/temporal_classifier/temporal_classifier.tflite"


@pytest.fixture()
def tflite_model_exists():
    """Skip if the dummy TFLite model hasn't been generated yet."""
    if not os.path.exists(TFLITE_PATH):
        pytest.skip(
            "Dummy TFLite model not found — run scripts/create_dummy_temporal_model.py first"
        )


# ---------------------------------------------------------------------------
# T2.1: Dummy TFLite model validation
# ---------------------------------------------------------------------------


class TestDummyTFLiteModel:
    """Verify the generated TFLite file has correct shapes and properties."""

    def test_tflite_file_exists(self, tflite_model_exists):
        assert os.path.isfile(TFLITE_PATH)

    def test_tflite_file_size_reasonable(self, tflite_model_exists):
        size_kb = os.path.getsize(TFLITE_PATH) / 1024
        # Design doc: ~75-123 KB.  Allow wider margin for quantization variance.
        assert 30 < size_kb < 200, f"Unexpected TFLite size: {size_kb:.1f} KB"

    def test_tflite_input_shape(self, tflite_model_exists):
        interp = Interpreter(model_path=TFLITE_PATH)
        interp.allocate_tensors()
        shape = tuple(interp.get_input_details()[0]["shape"])
        assert shape == (1, 20, 93)

    def test_tflite_output_shape(self, tflite_model_exists):
        interp = Interpreter(model_path=TFLITE_PATH)
        interp.allocate_tensors()
        shape = tuple(interp.get_output_details()[0]["shape"])
        assert shape == (1, 5)

    def test_tflite_softmax_sums_to_one(self, tflite_model_exists):
        interp = Interpreter(model_path=TFLITE_PATH)
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        out = interp.get_output_details()[0]

        dummy = np.random.randn(1, 20, 93).astype(np.float32)
        interp.set_tensor(inp["index"], dummy)
        interp.invoke()
        scores = np.squeeze(interp.get_tensor(out["index"]))
        assert abs(scores.sum() - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Integration: SequenceBuffer → TemporalClassifier (real TFLite)
# ---------------------------------------------------------------------------


class TestSequenceBufferToClassifier:
    """End-to-end: push frames into buffer, classify the window."""

    def test_full_pipeline(self, tflite_model_exists):
        from model.temporal_classifier import TemporalClassifier
        from utils.sequence_buffer import SequenceBuffer

        buf = SequenceBuffer(window_size=20, num_features=93)
        tc = TemporalClassifier(model_path=TFLITE_PATH)

        # Push 20 frames
        for _ in range(20):
            buf.push(np.random.randn(93).astype(np.float32))

        assert buf.is_ready()
        window = buf.get_window()
        assert window.shape == (20, 93)

        cls, scores = tc(window)
        assert 0 <= cls < 5
        assert len(scores) == 5
        assert abs(scores.sum() - 1.0) < 0.01

    def test_partial_buffer_still_classifies(self, tflite_model_exists):
        """Even with zero-padded partial buffer, classifier shouldn't crash."""
        from model.temporal_classifier import TemporalClassifier
        from utils.sequence_buffer import SequenceBuffer

        buf = SequenceBuffer(window_size=20, num_features=93)
        tc = TemporalClassifier(model_path=TFLITE_PATH)

        for _ in range(5):
            buf.push(np.random.randn(93).astype(np.float32))

        window = buf.get_window()
        cls, scores = tc(window)
        assert 0 <= cls < 5

    def test_classifier_properties(self, tflite_model_exists):
        from model.temporal_classifier import TemporalClassifier

        tc = TemporalClassifier(model_path=TFLITE_PATH)
        assert tc.window_size == 20
        assert tc.num_classes == 5
