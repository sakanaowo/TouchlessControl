"""Unit tests for Debouncer."""

import time
from unittest.mock import patch

import numpy as np
import pytest

from utils.debouncer import Debouncer
from utils.gesture_state_machine import GestureEvent


def _scores(class_id: int, confidence: float = 0.95, n: int = 5) -> np.ndarray:
    """Build a softmax-like score array with *confidence* at *class_id*."""
    rest = (1.0 - confidence) / max(n - 1, 1)
    s = np.full(n, rest, dtype=np.float32)
    s[class_id] = confidence
    return s


class TestDebouncerInit:
    def test_defaults(self):
        d = Debouncer()
        assert d.active_gesture is None
        assert d._confidence_threshold == 0.85
        assert d._cooldown == 0.3

    def test_custom(self):
        d = Debouncer(confidence_threshold=0.7, cooldown_seconds=0.1)
        assert d._confidence_threshold == 0.7
        assert d._cooldown == 0.1


class TestStartEvent:
    def test_non_null_triggers_start(self):
        d = Debouncer(cooldown_seconds=0.0)
        event = d.update(2, _scores(2))
        assert event == GestureEvent("left_click", "start")
        assert d.active_gesture == "left_click"

    def test_null_does_not_trigger(self):
        d = Debouncer()
        event = d.update(0, _scores(0))
        assert event is None
        assert d.active_gesture is None


class TestEndEvent:
    def test_active_to_null(self):
        d = Debouncer(cooldown_seconds=0.0)
        d.update(2, _scores(2))  # start
        event = d.update(0, _scores(0))
        assert event == GestureEvent("left_click", "end")
        assert d.active_gesture is None

    def test_gesture_switch_emits_end(self):
        d = Debouncer(cooldown_seconds=0.0)
        d.update(2, _scores(2))  # start left_click
        event = d.update(3, _scores(3))  # switch to drag_hold
        assert event == GestureEvent("left_click", "end")
        assert d.active_gesture is None


class TestHoldEvent:
    def test_same_gesture_hold(self):
        d = Debouncer(cooldown_seconds=0.0)
        d.update(1, _scores(1))  # start
        event = d.update(1, _scores(1))
        assert event == GestureEvent("pointer_move", "hold")

    @patch("utils.debouncer.time")
    def test_hold_respects_cooldown(self, mock_time):
        mock_time.time.side_effect = [1.0, 1.1, 1.35]
        d = Debouncer(cooldown_seconds=0.3)
        d.update(1, _scores(1))  # t=1.0: start (last_event=1.0)
        e2 = d.update(1, _scores(1))  # t=1.1: 0.1 < 0.3 → too soon
        assert e2 is None
        e3 = d.update(1, _scores(1))  # t=1.35: 0.35 >= 0.3 → hold
        assert e3 == GestureEvent("pointer_move", "hold")


class TestConfidenceThreshold:
    def test_low_confidence_forced_to_null(self):
        d = Debouncer(confidence_threshold=0.9, cooldown_seconds=0.0)
        # confidence 0.8 < threshold 0.9 → forced to null
        event = d.update(2, _scores(2, confidence=0.8))
        assert event is None

    def test_low_confidence_ends_active_gesture(self):
        d = Debouncer(confidence_threshold=0.9, cooldown_seconds=0.0)
        d.update(2, _scores(2, confidence=0.95))  # start
        event = d.update(2, _scores(2, confidence=0.8))  # below threshold → null → end
        assert event == GestureEvent("left_click", "end")


class TestUpdateNoHand:
    def test_no_hand_when_idle(self):
        d = Debouncer()
        event = d.update_no_hand()
        assert event is None

    def test_no_hand_ends_active(self):
        d = Debouncer(cooldown_seconds=0.0)
        d.update(3, _scores(3))
        event = d.update_no_hand()
        assert event == GestureEvent("drag_hold", "end")
        assert d.active_gesture is None


class TestGestureSwitchSequence:
    def test_switch_then_start_new(self):
        """A gesture switch emits 'end' first, then next frame can 'start' new."""
        d = Debouncer(cooldown_seconds=0.0)
        d.update(2, _scores(2))  # start left_click
        e1 = d.update(3, _scores(3))  # end left_click
        assert e1.event_type == "end"
        e2 = d.update(3, _scores(3))  # start drag_hold
        assert e2 == GestureEvent("drag_hold", "start")


class TestCooldownOnStart:
    @patch("utils.debouncer.time")
    def test_start_after_end_respects_cooldown(self, mock_time):
        mock_time.time.side_effect = [1.0, 1.5, 1.6]
        d = Debouncer(cooldown_seconds=0.3)
        d.update(2, _scores(2))  # t=1.0: start (last_event=1.0)
        d.update(0, _scores(0))  # t=1.5: end (last_event=1.5)
        e = d.update(2, _scores(2))  # t=1.6: 1.6-1.5=0.1 < 0.3 → no start
        assert e is None


class TestReset:
    def test_reset_clears_state(self):
        d = Debouncer(cooldown_seconds=0.0)
        d.update(2, _scores(2))
        assert d.active_gesture is not None
        d.reset()
        assert d.active_gesture is None
