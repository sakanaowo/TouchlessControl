import time
import unittest

import numpy as np

from utils.gesture_state_machine import GestureEvent, GestureStateMachine

NUM_CLASSES = 13


def _scores(class_id, confidence=0.90):
    """Build a fake softmax scores array with `confidence` at class_id."""
    s = np.zeros(NUM_CLASSES, dtype=np.float32)
    s[class_id] = confidence
    return s


class TestGestureStateMachineIdle(unittest.TestCase):
    def setUp(self):
        self.sm = GestureStateMachine()

    def test_stays_idle_on_null(self):
        for _ in range(10):
            event = self.sm.update(0, _scores(0))
        self.assertIsNone(event)
        self.assertEqual(self.sm.state, "idle")

    def test_stays_idle_below_threshold(self):
        for _ in range(10):
            event = self.sm.update(1, _scores(1, confidence=0.50))
        self.assertIsNone(event)
        self.assertEqual(self.sm.state, "idle")

    def test_no_hand_in_idle_is_noop(self):
        for _ in range(5):
            self.sm.update_no_hand()
        self.assertEqual(self.sm.state, "idle")


class TestGestureStateMachineTracking(unittest.TestCase):
    def setUp(self):
        self.sm = GestureStateMachine(activation_frames=5)

    def test_transitions_to_tracking_then_active(self):
        events = []
        for _ in range(5):
            e = self.sm.update(1, _scores(1))
            if e:
                events.append(e)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "start")
        self.assertEqual(events[0].gesture, "open_palm")
        self.assertEqual(self.sm.state, "active")

    def test_resets_to_idle_when_class_changes_before_activation(self):
        self.sm.update(1, _scores(1))
        self.sm.update(1, _scores(1))
        self.sm.update(2, _scores(2))  # class changes → reset
        self.assertEqual(self.sm.state, "idle")

    def test_no_hand_in_tracking_is_noop(self):
        self.sm.update(1, _scores(1))
        self.sm.update_no_hand()
        self.assertEqual(self.sm.state, "tracking")


class TestGestureStateMachineActive(unittest.TestCase):
    def _activate(self, sm, class_id=1):
        for _ in range(sm.activation_frames):
            sm.update(class_id, _scores(class_id))

    def test_emits_end_after_deactivation_frames(self):
        sm = GestureStateMachine(activation_frames=5, deactivation_frames=3)
        self._activate(sm, class_id=1)
        events = []
        for _ in range(3):
            e = sm.update(0, _scores(0))
            if e:
                events.append(e)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "end")
        self.assertEqual(sm.state, "idle")

    def test_debounce_prevents_immediate_reemit(self):
        sm = GestureStateMachine(activation_frames=5, debounce_seconds=10.0)
        self._activate(sm, class_id=1)
        # Try to emit hold — debounce is 10s, should not fire
        events = []
        for _ in range(5):
            e = sm.update(1, _scores(1))
            if e:
                events.append(e)
        self.assertEqual(len(events), 0)

    def test_hold_event_emitted_after_debounce(self):
        sm = GestureStateMachine(activation_frames=5, debounce_seconds=0.0)
        self._activate(sm, class_id=1)
        event = sm.update(1, _scores(1))
        self.assertIsNotNone(event)
        self.assertEqual(event.event_type, "hold")

    def test_no_hand_grace_period_preserves_active(self):
        sm = GestureStateMachine(activation_frames=5)
        self._activate(sm, class_id=1)
        # Simulate ~0.1s of no-hand frames (well within 1s grace)
        for _ in range(3):
            sm.update_no_hand()
            time.sleep(0.01)
        self.assertEqual(sm.state, "active")

    def test_no_hand_beyond_grace_period_resets_to_idle(self):
        sm = GestureStateMachine(activation_frames=5)
        self._activate(sm, class_id=1)
        # First call starts timer
        sm.update_no_hand()
        # Force timer to appear expired by manipulating internal timestamp
        sm._no_hand_since = time.time() - 1.1
        sm.update_no_hand()
        self.assertEqual(sm.state, "idle")

    def test_tracking_idle_no_hand_transition_unchanged(self):
        sm = GestureStateMachine(activation_frames=5)
        sm.update(1, _scores(1))
        sm.update(1, _scores(1))
        sm.update_no_hand()
        self.assertEqual(sm.state, "tracking")


if __name__ == "__main__":
    unittest.main()
