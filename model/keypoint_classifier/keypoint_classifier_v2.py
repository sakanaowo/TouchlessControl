import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite import Interpreter


class KeyPointClassifierV2:
    """
    TFLite classifier for the 93-dim v2 feature vector.
    Returns (class_index, softmax_scores) unlike v1 which returns class_index only.
    """

    def __init__(
        self,
        model_path="model/keypoint_classifier/keypoint_classifier_v2.tflite",
        num_threads=1,
    ):
        self.interpreter = Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, features):
        """
        Args:
            features: list or array of 93 float32 values

        Returns:
            (class_index: int, softmax_scores: np.ndarray)
        """
        self.interpreter.set_tensor(
            self.input_details[0]["index"],
            np.array([features], dtype=np.float32),
        )
        self.interpreter.invoke()

        scores = np.squeeze(
            self.interpreter.get_tensor(self.output_details[0]["index"])
        )
        class_index = int(np.argmax(scores))
        return class_index, scores
