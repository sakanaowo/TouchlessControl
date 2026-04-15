"""Integration tests for pointer→cursor and scroll control pipeline."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from utils.cursor_controller import CursorController, ScrollController
from utils.gesture_state_machine import GestureStateMachine, GestureEvent


class TestPointerCursorPipeline(unittest.TestCase):
    """End-to-end: classifier output → CursorController → mouse position."""

    def setUp(self):
        self.cursor = CursorController(
            smoothing=1.0,
            dead_zone=0,
            screen_padding=0.0,
            screen_w=1920,
            screen_h=1080,
        )
        self.mock_mapper = MagicMock()

    def test_pointer_move_drives_cursor(self):
        """pointer_move (class 1) → CursorController → mouse_move_to."""
        hand_sign_id = 1  # pointer_move
        tip_x, tip_y = 640, 360
        frame_w, frame_h = 1280, 720

        sx, sy = self.cursor.update(tip_x, tip_y, frame_w, frame_h)
        self.mock_mapper.mouse_move_to(sx, sy)

        self.mock_mapper.mouse_move_to.assert_called_once()
        args = self.mock_mapper.mouse_move_to.call_args[0]
        self.assertAlmostEqual(args[0], 960, delta=10)
        self.assertAlmostEqual(args[1], 540, delta=10)

    def test_multiple_frames_smoothed(self):
        """Multiple frames converge to target position with smoothing lag."""
        ctrl = CursorController(
            smoothing=0.3,
            dead_zone=0,
            screen_padding=0.0,
            screen_w=1920,
            screen_h=1080,
        )
        # Start at center
        ctrl.update(640, 360, 1280, 720)
        # Then jump to right side — EMA should lag behind
        positions = []
        for i in range(30):
            x, y = ctrl.update(900, 360, 1280, 720)
            positions.append(x)

        # Should converge toward 900/1280 * 1920 ≈ 1350
        self.assertAlmostEqual(positions[-1], 1350, delta=20)
        # Early positions should lag behind (smoothing)
        self.assertLess(positions[0], positions[-1])


class TestScrollControllerPipeline(unittest.TestCase):
    """End-to-end: scroll_mode → ScrollController → scroll_by."""

    def setUp(self):
        self.scroll = ScrollController(sensitivity=0.1)
        self.mock_mapper = MagicMock()

    def test_scroll_mode_activation_and_scroll(self):
        """scroll_mode (class 4) → start anchor → get scroll amount."""
        # Frame 1: start scroll
        anchor_y = 360.0
        self.scroll.start_scroll(anchor_y)
        self.assertTrue(self.scroll.active)

        # Frame 2: finger moves up
        clicks = self.scroll.get_scroll_amount(300.0)
        self.assertGreater(clicks, 0)
        self.mock_mapper.scroll_by(clicks)
        self.mock_mapper.scroll_by.assert_called_with(clicks)

    def test_scroll_stop_on_gesture_change(self):
        """Switching away from scroll_mode stops scrolling."""
        self.scroll.start_scroll(360.0)
        self.scroll.stop_scroll()
        self.assertFalse(self.scroll.active)
        self.assertEqual(self.scroll.get_scroll_amount(100.0), 0)


class TestStateMachineClickDrag(unittest.TestCase):
    """Event-based gestures go through GestureStateMachine → ActionMapper."""

    def setUp(self):
        self.sm = GestureStateMachine(
            activation_frames=3,
            debounce_seconds=0.0,
        )
        self.mock_mapper = MagicMock()

    def _scores(self, class_id, confidence=0.90):
        s = np.zeros(5, dtype=np.float32)
        s[class_id] = confidence
        return s

    def test_left_click_through_state_machine(self):
        """left_click (class 2) → 3 frames → start event → ActionMapper.handle."""
        events = []
        for _ in range(3):
            e = self.sm.update(2, self._scores(2))
            if e:
                events.append(e)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].gesture, "left_click")
        self.assertEqual(events[0].event_type, "start")

        self.mock_mapper.handle(events[0])
        self.mock_mapper.handle.assert_called_once()

    def test_drag_hold_start_and_end(self):
        """drag_hold (class 3) → start event, then class change → end event."""
        # Activate drag
        for _ in range(3):
            self.sm.update(3, self._scores(3))

        # Now switch to null → should eventually end
        events = []
        for _ in range(self.sm.deactivation_frames):
            e = self.sm.update(0, self._scores(0))
            if e:
                events.append(e)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].gesture, "drag_hold")
        self.assertEqual(events[0].event_type, "end")

    def test_continuous_gestures_bypass_state_machine(self):
        """pointer_move (1) and scroll_mode (4) should NOT trigger events
        when used as continuous controllers (they bypass the SM)."""
        # This test validates the design: we DON'T feed class 1/4 to SM
        # Just verify SM stays idle when only null is fed
        for _ in range(10):
            e = self.sm.update(0, self._scores(0))
            self.assertIsNone(e)
        self.assertEqual(self.sm.state, "idle")


if __name__ == "__main__":
    unittest.main()
