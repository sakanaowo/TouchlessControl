import time
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

GESTURE_LABELS = [
    "null",  # 0
    "open_palm",  # 1
    "fist",  # 2
    "pointer",  # 3
    "thumbs_up",  # 4
    "thumbs_down",  # 5
    "v_sign",  # 6
    "three_fingers",  # 7
    "four_fingers",  # 8
    "pinch",  # 9
    "ok_sign",  # 10
    "gun_sign",  # 11
    "call_sign",  # 12
]


@dataclass
class GestureEvent:
    gesture: str
    event_type: Literal["start", "end", "hold"]


@dataclass
class GestureStateMachine:
    confidence_threshold: float = 0.82
    activation_frames: int = 5
    deactivation_frames: int = 10
    debounce_seconds: float = 0.5

    _state: str = field(default="idle", init=False, repr=False)
    _candidate_class: int = field(default=0, init=False, repr=False)
    _candidate_frames: int = field(default=0, init=False, repr=False)
    _deactivation_frames_count: int = field(default=0, init=False, repr=False)
    _active_class: int = field(default=-1, init=False, repr=False)
    _last_emitted_time: float = field(default=0.0, init=False, repr=False)
    _no_hand_since: float = field(default=0.0, init=False, repr=False)

    @property
    def state(self) -> str:
        return self._state

    @property
    def active_gesture(self) -> Optional[str]:
        if self._active_class >= 0:
            return GESTURE_LABELS[self._active_class]
        return None

    def update(self, class_id: int, scores: np.ndarray) -> Optional[GestureEvent]:
        """
        Call each frame when hand landmarks ARE detected.

        Args:
            class_id: argmax class index from classifier
            scores:   full softmax probability array (shape: [num_classes])

        Returns:
            GestureEvent or None
        """
        confidence = float(scores[class_id])
        is_null = class_id == 0
        is_confident = confidence >= self.confidence_threshold
        now = time.time()
        self._no_hand_since = 0.0  # hand present this frame

        if self._state == "idle":
            if not is_null and is_confident:
                self._state = "tracking"
                self._candidate_class = class_id
                self._candidate_frames = 1
            return None

        if self._state == "tracking":
            if class_id == self._candidate_class and is_confident:
                self._candidate_frames += 1
                if self._candidate_frames >= self.activation_frames:
                    if now - self._last_emitted_time > self.debounce_seconds:
                        self._state = "active"
                        self._active_class = class_id
                        self._deactivation_frames_count = 0
                        self._last_emitted_time = now
                        return GestureEvent(GESTURE_LABELS[class_id], "start")
            else:
                self._state = "idle"
                self._candidate_frames = 0
            return None

        if self._state == "active":
            if class_id != self._active_class or not is_confident:
                self._deactivation_frames_count += 1
                if self._deactivation_frames_count >= self.deactivation_frames:
                    ended = self._active_class
                    self._state = "idle"
                    self._active_class = -1
                    self._deactivation_frames_count = 0
                    return GestureEvent(GESTURE_LABELS[ended], "end")
            else:
                self._deactivation_frames_count = 0
                if now - self._last_emitted_time >= self.debounce_seconds:
                    self._last_emitted_time = now
                    return GestureEvent(GESTURE_LABELS[class_id], "hold")
            return None

        return None

    def update_no_hand(self) -> None:
        """
        Call each frame when no hand landmarks are detected.
        Preserves ACTIVE state for a 1-second grace period.
        After 1 s without a hand, silently resets to IDLE.
        IDLE / TRACKING states are unaffected.
        """
        if self._state != "active":
            return

        now = time.time()
        if self._no_hand_since == 0.0:
            self._no_hand_since = now
        elif now - self._no_hand_since >= 1.0:
            self._state = "idle"
            self._active_class = -1
            self._deactivation_frames_count = 0
            self._no_hand_since = 0.0
