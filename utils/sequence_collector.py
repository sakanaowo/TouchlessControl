"""Continuous sequence data collector for temporal model training.

Records frames into a RAM buffer during a session, then extracts
overlapping sliding windows and appends them to a compressed NPZ file.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np


class SequenceCollector:
    """Record continuous gesture sessions and persist as sliding-window sequences.

    Parameters
    ----------
    window_size:
        Number of frames per sequence (must match model input).
    stride:
        Step size between consecutive windows during extraction.
        ``stride=5`` at 30 fps ≈ 6 sequences per second of recording.
    num_features:
        Feature dimensionality per frame.
    save_path:
        Path to the ``.npz`` file.  Created on first :meth:`stop_recording`.
    """

    def __init__(
        self,
        window_size: int = 20,
        stride: int = 5,
        num_features: int = 93,
        save_path: str = "model/temporal_classifier/keypoint_sequences.npz",
    ) -> None:
        self._window_size = window_size
        self._stride = stride
        self._num_features = num_features
        self._save_path = save_path

        self._recording_buffer: list[np.ndarray] = []
        self._class_id: int = -1
        self._is_recording: bool = False

    # ------------------------------------------------------------------ #
    # Recording lifecycle
    # ------------------------------------------------------------------ #

    def start_recording(self, class_id: int) -> None:
        """Begin a new recording session for *class_id*."""
        self._class_id = class_id
        self._recording_buffer = []
        self._is_recording = True

    def add_frame(self, features: list[float] | np.ndarray) -> None:
        """Append one frame.  No-op when not recording."""
        if not self._is_recording:
            return
        self._recording_buffer.append(np.asarray(features, dtype=np.float32))

    def stop_recording(self) -> int:
        """Stop recording, extract windows, and append to NPZ.

        Returns the number of new sequences saved (0 if the session was
        too short to form even one window).
        """
        if not self._is_recording:
            return 0
        self._is_recording = False

        frames = np.array(self._recording_buffer, dtype=np.float32)
        self._recording_buffer = []
        num_frames = len(frames)

        if num_frames < self._window_size:
            return 0

        # Sliding window extraction
        sequences: list[np.ndarray] = []
        for start in range(0, num_frames - self._window_size + 1, self._stride):
            sequences.append(frames[start : start + self._window_size])

        new_sequences = np.array(sequences, dtype=np.float32)
        new_labels = np.full(len(sequences), self._class_id, dtype=np.int32)

        # Append to existing data
        existing = self._load_npz()
        if existing is not None:
            all_sequences = np.concatenate([existing["sequences"], new_sequences])
            all_labels = np.concatenate([existing["labels"], new_labels])
        else:
            all_sequences = new_sequences
            all_labels = new_labels

        os.makedirs(os.path.dirname(self._save_path) or ".", exist_ok=True)
        np.savez_compressed(self._save_path, sequences=all_sequences, labels=all_labels)
        return len(sequences)

    def cancel_recording(self) -> None:
        """Discard the current session buffer without saving."""
        self._is_recording = False
        self._recording_buffer = []

    # ------------------------------------------------------------------ #
    # Query helpers
    # ------------------------------------------------------------------ #

    def load_existing(self) -> dict[int, int]:
        """Return ``{class_id: count}`` from the persisted NPZ."""
        data = self._load_npz()
        if data is None:
            return {}
        labels = data["labels"]
        return {int(lbl): int(np.sum(labels == lbl)) for lbl in np.unique(labels)}

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def frames_recorded(self) -> int:
        return len(self._recording_buffer)

    # ------------------------------------------------------------------ #

    def _load_npz(self) -> Optional[np.lib.npyio.NpzFile]:
        if os.path.exists(self._save_path):
            return np.load(self._save_path)
        return None
