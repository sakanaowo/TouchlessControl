"""Unit tests for TemporalClassifier (mocked TFLite interpreter)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_mock_interpreter(window_size=20, num_features=93, num_classes=5):
    """Build a MagicMock that mimics a TFLite Interpreter for TemporalClassifier."""
    interp = MagicMock()
    interp.get_input_details.return_value = [
        {"index": 0, "shape": np.array([1, window_size, num_features])}
    ]
    interp.get_output_details.return_value = [
        {"index": 1, "shape": np.array([1, num_classes])}
    ]

    # Default: return uniform distribution
    def _get_tensor(index):
        if index == 1:
            return np.full((1, num_classes), 1.0 / num_classes, dtype=np.float32)
        return None

    interp.get_tensor.side_effect = _get_tensor
    return interp


@pytest.fixture()
def classifier():
    """Return a TemporalClassifier with a mocked interpreter."""
    with patch(
        "model.temporal_classifier.temporal_classifier.Interpreter"
    ) as MockInterp:
        mock_interp = _make_mock_interpreter()
        MockInterp.return_value = mock_interp

        from model.temporal_classifier.temporal_classifier import (
            TemporalClassifier,
        )

        tc = TemporalClassifier(model_path="dummy.tflite")
    tc._mock = mock_interp  # expose for assertions
    return tc


class TestTemporalClassifierInit:
    def test_properties(self, classifier):
        assert classifier.window_size == 20
        assert classifier.num_classes == 5


class TestTemporalClassifierCall:
    def test_2d_input(self, classifier):
        """Input (W, F) is auto-expanded to (1, W, F)."""
        window = np.zeros((20, 93), dtype=np.float32)
        class_id, scores = classifier(window)
        assert isinstance(class_id, int)
        assert scores.shape == (5,)
        np.testing.assert_allclose(scores.sum(), 1.0, atol=1e-5)

    def test_3d_input(self, classifier):
        """Input (1, W, F) works directly."""
        window = np.zeros((1, 20, 93), dtype=np.float32)
        class_id, scores = classifier(window)
        assert isinstance(class_id, int)
        assert scores.shape == (5,)

    def test_set_tensor_called_with_correct_shape(self, classifier):
        window = np.zeros((20, 93), dtype=np.float32)
        classifier(window)
        call_args = classifier._mock.set_tensor.call_args
        tensor = call_args[0][1]
        assert tensor.shape == (1, 20, 93)
        assert tensor.dtype == np.float32

    def test_argmax_class(self, classifier):
        """When output has a clear peak, class_id matches argmax."""
        scores_arr = np.array([[0.05, 0.05, 0.8, 0.05, 0.05]], dtype=np.float32)
        classifier._mock.get_tensor.side_effect = lambda idx: (
            scores_arr if idx == 1 else None
        )
        class_id, scores = classifier(np.zeros((20, 93)))
        assert class_id == 2
        assert float(scores[2]) == pytest.approx(0.8)

    def test_invoke_called(self, classifier):
        classifier(np.zeros((20, 93)))
        classifier._mock.invoke.assert_called_once()
