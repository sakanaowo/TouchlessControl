#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    # Fallback for environments that still use TensorFlow Lite.
    from tensorflow.lite import Interpreter


class KeyPointClassifier(object):
    def __init__(
        self,
        model_path="model/keypoint_classifier/keypoint_classifier.tflite",
        num_threads=1,
    ):
        self.interpreter = Interpreter(model_path=model_path, num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(
        self,
        landmark_list,
    ):
        input_details_tensor_index = self.input_details[0]["index"]
        self.interpreter.set_tensor(
            input_details_tensor_index, np.array([landmark_list], dtype=np.float32)
        )
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]["index"]

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        return result_index
