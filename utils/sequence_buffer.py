"""Ring buffer holding the W most recent feature frames for temporal classification."""

from __future__ import annotations

import numpy as np


class SequenceBuffer:
    """Circular buffer that stores the last *window_size* feature vectors.

    Each call to :meth:`push` overwrites the oldest frame once the buffer is
    full.  :meth:`get_window` always returns an array of shape
    ``(window_size, num_features)`` in chronological order (oldest → newest),
    zero-padded at the front when fewer than *window_size* frames have been
    pushed.
    """

    def __init__(self, window_size: int = 20, num_features: int = 93) -> None:
        self._window_size = window_size
        self._num_features = num_features
        self._buffer = np.zeros((window_size, num_features), dtype=np.float32)
        self._count: int = 0  # total frames pushed since last clear

    # --- mutators --------------------------------------------------------- #

    def push(self, features: list[float] | np.ndarray) -> None:
        """Append one frame to the buffer (circular overwrite when full)."""
        pos = self._count % self._window_size
        self._buffer[pos] = np.asarray(features, dtype=np.float32)
        self._count += 1

    def clear(self) -> None:
        """Reset the buffer to all zeros and rewind the counter."""
        self._buffer[:] = 0.0
        self._count = 0

    # --- accessors -------------------------------------------------------- #

    def get_window(self) -> np.ndarray:
        """Return the current window as a *new* array ``(W, F)``.

        - If ``count >= window_size``: chronological order, no padding.
        - If ``0 < count < window_size``: zero-padded at the front.
        - If ``count == 0``: all zeros.
        """
        if self._count == 0:
            return np.zeros((self._window_size, self._num_features), dtype=np.float32)

        if self._count >= self._window_size:
            start = self._count % self._window_size
            if start == 0:
                return self._buffer.copy()
            return np.concatenate([self._buffer[start:], self._buffer[:start]], axis=0)

        # Partial fill — zero-pad at the front
        result = np.zeros((self._window_size, self._num_features), dtype=np.float32)
        result[-self._count :] = self._buffer[: self._count]
        return result

    def is_ready(self) -> bool:
        """``True`` when at least *window_size* frames have been pushed."""
        return self._count >= self._window_size

    @property
    def count(self) -> int:
        """Total frames pushed since the last :meth:`clear`."""
        return self._count

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def num_features(self) -> int:
        return self._num_features
