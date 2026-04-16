"""Lightweight event debouncer replacing the full GestureStateMachine.

The GRU-64 temporal model already handles temporal smoothing, activation
detection, and transition noise.  The Debouncer only needs to:

1. Apply a confidence threshold.
2. Detect edges (null ↔ non-null) and emit start / end / hold events.
3. Enforce a cooldown between repeated ``hold`` events.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from utils.gesture_state_machine import GESTURE_LABELS, GestureEvent


class Debouncer:
    """Emit :class:`GestureEvent` objects based on classifier output.

    Parameters
    ----------
    confidence_threshold:
        Minimum ``scores[class_id]`` to accept a prediction; otherwise
        the class is forced to *null* (index 0).
    cooldown_seconds:
        Minimum interval between consecutive ``hold`` events for the
        same gesture.  ``start`` and ``end`` are emitted immediately.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.85,
        cooldown_seconds: float = 0.3,
    ) -> None:
        self._confidence_threshold = confidence_threshold
        self._cooldown = cooldown_seconds
        self._active_gesture: str | None = None
        self._active_class: int = -1
        self._last_event_time: float = 0.0

    # ------------------------------------------------------------------ #

    def update(self, class_id: int, scores: np.ndarray) -> Optional[GestureEvent]:
        """Process one frame of classifier output.

        Returns a :class:`GestureEvent` when a meaningful transition or
        hold tick occurs, otherwise ``None``.
        """
        confidence = float(scores[class_id])
        now = time.time()

        # Reject low-confidence → treat as null
        if confidence < self._confidence_threshold:
            class_id = 0

        is_null = class_id == 0

        # ---- currently idle ---- #
        if self._active_gesture is None:
            if not is_null:
                if now - self._last_event_time >= self._cooldown:
                    name = GESTURE_LABELS[class_id]
                    self._active_gesture = name
                    self._active_class = class_id
                    self._last_event_time = now
                    return GestureEvent(name, "start")
            return None

        # ---- currently active ---- #
        if is_null:
            ended = self._active_gesture
            self._active_gesture = None
            self._active_class = -1
            self._last_event_time = now
            return GestureEvent(ended, "end")

        if class_id == self._active_class:
            # Same gesture continues
            if now - self._last_event_time >= self._cooldown:
                self._last_event_time = now
                return GestureEvent(self._active_gesture, "hold")
            return None

        # Different gesture → end old; new gesture starts next frame
        ended = self._active_gesture
        self._active_gesture = None
        self._active_class = -1
        self._last_event_time = now
        return GestureEvent(ended, "end")

    def update_no_hand(self) -> Optional[GestureEvent]:
        """Call when no hand is detected — immediately ends current gesture."""
        if self._active_gesture is None:
            return None
        ended = self._active_gesture
        self._active_gesture = None
        self._active_class = -1
        self._last_event_time = time.time()
        return GestureEvent(ended, "end")

    # ------------------------------------------------------------------ #

    @property
    def active_gesture(self) -> str | None:
        """Currently active gesture name, or ``None``."""
        return self._active_gesture

    def reset(self) -> None:
        """Silently reset to idle without emitting events."""
        self._active_gesture = None
        self._active_class = -1
        self._last_event_time = 0.0
