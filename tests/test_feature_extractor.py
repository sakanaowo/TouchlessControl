import unittest

import numpy as np

from utils.feature_extractor import FeatureExtractor


class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = FeatureExtractor()

    def test_extract_returns_93_features_with_3d_input(self):
        landmarks = np.random.rand(21, 3).astype(np.float32)
        features = self.extractor.extract(landmarks)

        self.assertEqual(len(features), 93)
        self.assertFalse(np.isnan(np.array(features, dtype=np.float32)).any())

    def test_extract_returns_93_features_with_2d_input(self):
        landmarks = np.random.rand(21, 2).astype(np.float32)
        features = self.extractor.extract(landmarks)

        self.assertEqual(len(features), 93)
        self.assertFalse(np.isnan(np.array(features, dtype=np.float32)).any())

    def test_extract_legacy_xy_returns_42_features(self):
        landmarks = np.random.rand(21, 3).astype(np.float32)
        features = self.extractor.extract_legacy_xy(landmarks)

        self.assertEqual(len(features), 42)
        self.assertFalse(np.isnan(np.array(features, dtype=np.float32)).any())


if __name__ == "__main__":
    unittest.main()
