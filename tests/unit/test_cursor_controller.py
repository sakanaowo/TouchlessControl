"""Unit tests for CursorController and ScrollController."""

import unittest

from utils.cursor_controller import CursorController, ScrollController


class TestCursorControllerInit(unittest.TestCase):
    def test_default_screen_fallback(self):
        """When screeninfo is unavailable, falls back to 1920x1080."""
        ctrl = CursorController(screen_w=1920, screen_h=1080)
        self.assertEqual(ctrl._screen_w, 1920)
        self.assertEqual(ctrl._screen_h, 1080)

    def test_custom_screen_size(self):
        ctrl = CursorController(screen_w=2560, screen_h=1440)
        self.assertEqual(ctrl._screen_w, 2560)
        self.assertEqual(ctrl._screen_h, 1440)


class TestCursorControllerUpdate(unittest.TestCase):
    def setUp(self):
        self.ctrl = CursorController(
            smoothing=0.3,
            dead_zone=5,
            screen_padding=0.0,
            screen_w=1920,
            screen_h=1080,
        )

    def test_center_maps_to_screen_center(self):
        x, y = self.ctrl.update(640, 360, 1280, 720)
        self.assertAlmostEqual(x, 960, delta=10)
        self.assertAlmostEqual(y, 540, delta=10)

    def test_top_left_maps_near_zero(self):
        x, y = self.ctrl.update(0, 0, 1280, 720)
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_bottom_right_maps_near_max(self):
        x, y = self.ctrl.update(1280, 720, 1280, 720)
        self.assertAlmostEqual(x, 1919, delta=1)
        self.assertAlmostEqual(y, 1079, delta=1)

    def test_clamping_negative_coords(self):
        x, y = self.ctrl.update(-100, -100, 1280, 720)
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_clamping_over_frame(self):
        x, y = self.ctrl.update(2000, 1500, 1280, 720)
        self.assertAlmostEqual(x, 1919, delta=1)
        self.assertAlmostEqual(y, 1079, delta=1)


class TestCursorControllerSmoothing(unittest.TestCase):
    def test_ema_convergence(self):
        """After many frames at same position, output converges."""
        ctrl = CursorController(
            smoothing=0.3,
            dead_zone=0,
            screen_padding=0.0,
            screen_w=1920,
            screen_h=1080,
        )
        for _ in range(50):
            x, y = ctrl.update(640, 360, 1280, 720)
        self.assertAlmostEqual(x, 960, delta=2)
        self.assertAlmostEqual(y, 540, delta=2)

    def test_first_frame_no_smoothing(self):
        """First frame should pass through without smoothing."""
        ctrl = CursorController(
            smoothing=0.3,
            dead_zone=0,
            screen_padding=0.0,
            screen_w=1920,
            screen_h=1080,
        )
        x, y = ctrl.update(640, 360, 1280, 720)
        self.assertAlmostEqual(x, 960, delta=1)
        self.assertAlmostEqual(y, 540, delta=1)


class TestCursorControllerDeadZone(unittest.TestCase):
    def test_small_movement_filtered(self):
        """Movement below dead_zone threshold returns previous position."""
        ctrl = CursorController(
            smoothing=1.0,
            dead_zone=20,
            screen_padding=0.0,
            screen_w=1920,
            screen_h=1080,
        )
        x1, y1 = ctrl.update(640, 360, 1280, 720)
        # Tiny movement: 1 pixel in camera space → ~1.5px screen
        x2, y2 = ctrl.update(641, 360, 1280, 720)
        self.assertEqual(x1, x2)
        self.assertEqual(y1, y2)

    def test_large_movement_passes(self):
        """Movement above dead_zone goes through."""
        ctrl = CursorController(
            smoothing=1.0,
            dead_zone=5,
            screen_padding=0.0,
            screen_w=1920,
            screen_h=1080,
        )
        ctrl.update(640, 360, 1280, 720)
        x2, y2 = ctrl.update(700, 360, 1280, 720)
        # 60px camera → ~90px screen at 1920/1280 ratio
        self.assertGreater(x2, 960)


class TestCursorControllerPadding(unittest.TestCase):
    def test_padding_expands_usable_area(self):
        """With padding, camera edge maps beyond [0,1] → clamped."""
        ctrl = CursorController(
            smoothing=1.0,
            dead_zone=0,
            screen_padding=0.1,
            screen_w=1920,
            screen_h=1080,
        )
        # At 10% padding, finger at x=0 maps to (-0.1/0.8) < 0 → clamped to 0
        x, y = ctrl.update(0, 0, 1280, 720)
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

        # Finger at center should still map near center
        ctrl.reset()
        x, y = ctrl.update(640, 360, 1280, 720)
        self.assertAlmostEqual(x, 960, delta=10)
        self.assertAlmostEqual(y, 540, delta=10)


class TestCursorControllerReset(unittest.TestCase):
    def test_reset_clears_state(self):
        ctrl = CursorController(
            smoothing=0.3,
            dead_zone=0,
            screen_padding=0.0,
            screen_w=1920,
            screen_h=1080,
        )
        ctrl.update(100, 100, 1280, 720)
        ctrl.reset()
        self.assertIsNone(ctrl._prev_x)
        self.assertIsNone(ctrl._prev_y)

    def test_after_reset_no_smoothing(self):
        """After reset, next frame acts as first frame (no EMA history)."""
        ctrl = CursorController(
            smoothing=0.3,
            dead_zone=0,
            screen_padding=0.0,
            screen_w=1920,
            screen_h=1080,
        )
        ctrl.update(100, 100, 1280, 720)
        ctrl.reset()
        x, y = ctrl.update(640, 360, 1280, 720)
        self.assertAlmostEqual(x, 960, delta=1)
        self.assertAlmostEqual(y, 540, delta=1)


# ── ScrollController Tests ──────────────────────────────────────────


class TestScrollControllerInit(unittest.TestCase):
    def test_not_active_initially(self):
        sc = ScrollController()
        self.assertFalse(sc.active)

    def test_zero_scroll_when_inactive(self):
        sc = ScrollController()
        self.assertEqual(sc.get_scroll_amount(100), 0)


class TestScrollControllerActive(unittest.TestCase):
    def setUp(self):
        self.sc = ScrollController(sensitivity=0.1)

    def test_start_sets_active(self):
        self.sc.start_scroll(300.0)
        self.assertTrue(self.sc.active)

    def test_no_movement_returns_zero(self):
        self.sc.start_scroll(300.0)
        self.assertEqual(self.sc.get_scroll_amount(300.0), 0)

    def test_finger_up_scrolls_positive(self):
        """Finger moves UP (smaller Y) → scroll up (positive)."""
        self.sc.start_scroll(300.0)
        clicks = self.sc.get_scroll_amount(200.0)  # 100px up
        self.assertGreater(clicks, 0)

    def test_finger_down_scrolls_negative(self):
        """Finger moves DOWN (larger Y) → scroll down (negative)."""
        self.sc.start_scroll(300.0)
        clicks = self.sc.get_scroll_amount(400.0)  # 100px down
        self.assertLess(clicks, 0)

    def test_scroll_proportional_to_displacement(self):
        self.sc.start_scroll(300.0)
        small = self.sc.get_scroll_amount(280.0)  # 20px
        self.sc.start_scroll(300.0)
        large = self.sc.get_scroll_amount(200.0)  # 100px
        self.assertGreater(abs(large), abs(small))

    def test_stop_resets_active(self):
        self.sc.start_scroll(300.0)
        self.sc.stop_scroll()
        self.assertFalse(self.sc.active)

    def test_after_stop_returns_zero(self):
        self.sc.start_scroll(300.0)
        self.sc.stop_scroll()
        self.assertEqual(self.sc.get_scroll_amount(200.0), 0)


if __name__ == "__main__":
    unittest.main()
