"""TFLite wrapper for the GRU-64 temporal gesture classifier."""

from __future__ import annotations

import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite import Interpreter


class TemporalClassifier:
    """Run inference on a temporal (sequence) TFLite model.

    The expected input shape is ``(1, window_size, num_features)`` and the
    output is ``(1, num_classes)`` with softmax probabilities.

    The :meth:`__call__` interface mirrors :class:`KeyPointClassifierV2`:
    it returns ``(class_index, softmax_scores)``.
    """

    def __init__(
        self,
        model_path: str | None = None,
        num_threads: int = 1,
    ) -> None:
        if model_path is None:
            model_path = "model/temporal_classifier/temporal_classifier.tflite"

        self.interpreter = Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, window: np.ndarray) -> tuple[int, np.ndarray]:
        """Classify a sequence window.

        Parameters
        ----------
        window:
            Array of shape ``(W, F)`` or ``(1, W, F)``.

        Returns
        -------
        (class_index, softmax_scores)
        """
        input_array = np.asarray(window, dtype=np.float32)
        if input_array.ndim == 2:
            input_array = np.expand_dims(input_array, axis=0)

        self.interpreter.set_tensor(self.input_details[0]["index"], input_array)
        self.interpreter.invoke()

        scores = np.squeeze(
            self.interpreter.get_tensor(self.output_details[0]["index"])
        )
        class_index = int(np.argmax(scores))
        return class_index, scores

    @property
    def window_size(self) -> int:
        """Expected number of timesteps (``W``) from the model input shape."""
        return int(self.input_details[0]["shape"][1])

    @property
    def num_classes(self) -> int:
        """Number of output classes from the model output shape."""
        return int(self.output_details[0]["shape"][-1])
