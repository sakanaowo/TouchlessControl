import numpy as np


class FeatureExtractor:
    """Extract a 93-dim hand feature vector from 21 hand landmarks."""

    PALM_KP_INDICES = [0, 5, 9, 13, 17]
    FINGER_JOINTS = [
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20],
        [0, 1, 2, 3, 4],
    ]

    def extract(self, landmark_list):
        """
        Args:
            landmark_list: list of 21 points, each point as [x, y] or [x, y, z]

        Returns:
            list[float]: 93 features
        """
        kp = self._to_21x3_array(landmark_list)

        # Block 1: 3D coordinates normalized relative to wrist.
        kp_rel = kp - kp[0]
        max_dist = float(np.max(np.linalg.norm(kp_rel, axis=1)))
        scale = max(max_dist, 1e-6)
        kp_norm = (kp_rel / scale).flatten().tolist()  # 63

        # Block 2: 15 joint-angle cosine values.
        angles = self._compute_joint_angles(kp)

        # Block 3: 5 fingertip-to-wrist distances.
        tips = [4, 8, 12, 16, 20]
        tip_wrist = [float(np.linalg.norm(kp[t] - kp[0]) / scale) for t in tips]

        # Block 4: 5 fingertip-to-palm-center distances.
        palm_center = kp[self.PALM_KP_INDICES].mean(axis=0)
        tip_palm = [float(np.linalg.norm(kp[t] - palm_center) / scale) for t in tips]

        # Block 5: 5 finger-state booleans (extended or bent).
        finger_states = self._compute_finger_states(kp)

        features = kp_norm + angles + tip_wrist + tip_palm + finger_states
        if len(features) != 93:
            raise ValueError(f"Expected 93 features, got {len(features)}")
        return features

    def extract_legacy_xy(self, landmark_list):
        """Return legacy 42-dim XY features for the existing v1 classifier."""
        kp = self._to_21x3_array(landmark_list)
        kp_xy = kp[:, :2].copy()
        kp_xy -= kp_xy[0]

        flattened = kp_xy.flatten()
        max_value = float(np.max(np.abs(flattened)))
        if max_value < 1e-6:
            max_value = 1.0
        return (flattened / max_value).astype(np.float32).tolist()

    def _to_21x3_array(self, landmark_list):
        kp = np.asarray(landmark_list, dtype=np.float32)
        if kp.shape[0] != 21:
            raise ValueError(f"Expected 21 keypoints, got shape {kp.shape}")

        if kp.ndim != 2:
            raise ValueError(f"Expected 2D array-like input, got {kp.ndim}D")

        if kp.shape[1] == 2:
            kp = np.concatenate([kp, np.zeros((21, 1), dtype=np.float32)], axis=1)
        elif kp.shape[1] != 3:
            raise ValueError("Each keypoint must have 2 or 3 values")

        return kp

    def _compute_joint_angles(self, kp):
        angles = []
        for joints in self.FINGER_JOINTS:
            for idx in range(len(joints) - 2):
                a = kp[joints[idx]]
                b = kp[joints[idx + 1]]
                c = kp[joints[idx + 2]]
                v1 = a - b
                v2 = c - b
                denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
                cosine = float(np.dot(v1, v2) / denom)
                angles.append(float(np.clip(cosine, -1.0, 1.0)))
        return angles

    def _compute_finger_states(self, kp):
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        wrist = kp[0]

        states = []
        for tip, pip in zip(tips, pips):
            tip_dist = np.linalg.norm(kp[tip] - wrist)
            pip_dist = np.linalg.norm(kp[pip] - wrist)
            states.append(1.0 if tip_dist > pip_dist else 0.0)

        thumb_tip_dist = np.linalg.norm(kp[4] - wrist)
        thumb_mcp_dist = np.linalg.norm(kp[2] - wrist)
        states.append(1.0 if thumb_tip_dist > thumb_mcp_dist else 0.0)
        return states
