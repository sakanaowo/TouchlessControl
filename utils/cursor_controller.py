"""CursorController and ScrollController for touchless mouse control."""

from typing import Optional, Tuple


class CursorController:
    """Map hand landmark position to screen cursor with EMA smoothing.

    Camera frame coordinates → screen coordinates with padding margin,
    exponential moving average smoothing, and dead zone filtering.
    """

    def __init__(
        self,
        smoothing: float = 0.3,
        dead_zone: int = 5,
        screen_padding: float = 0.08,
        screen_w: Optional[int] = None,
        screen_h: Optional[int] = None,
    ):
        self._alpha = smoothing
        self._dead_zone = dead_zone
        self._padding = screen_padding

        # Auto-detect screen resolution if not provided
        if screen_w is not None and screen_h is not None:
            self._screen_w = screen_w
            self._screen_h = screen_h
        else:
            self._screen_w, self._screen_h = self._detect_screen()

        self._prev_x: Optional[float] = None
        self._prev_y: Optional[float] = None

    @staticmethod
    def _detect_screen() -> Tuple[int, int]:
        """Detect primary monitor resolution via screeninfo."""
        try:
            from screeninfo import get_monitors

            m = get_monitors()[0]
            return m.width, m.height
        except Exception:
            return 1920, 1080  # safe fallback

    def update(
        self, finger_x: float, finger_y: float, frame_w: int, frame_h: int
    ) -> Tuple[int, int]:
        """Convert camera-space finger coords to smoothed screen coords.

        Args:
            finger_x: Pixel X of fingertip in camera frame.
            finger_y: Pixel Y of fingertip in camera frame.
            frame_w: Camera frame width.
            frame_h: Camera frame height.

        Returns:
            (screen_x, screen_y) — clamped to screen bounds.
        """
        # Normalize to [0, 1] with padding margin
        pad = self._padding
        nx = (finger_x / frame_w - pad) / (1.0 - 2 * pad)
        ny = (finger_y / frame_h - pad) / (1.0 - 2 * pad)

        # Clamp to [0, 1]
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))

        # Map to screen coordinates
        raw_x = nx * self._screen_w
        raw_y = ny * self._screen_h

        # EMA smoothing
        if self._prev_x is None:
            sx, sy = raw_x, raw_y
        else:
            sx = self._alpha * raw_x + (1 - self._alpha) * self._prev_x
            sy = self._alpha * raw_y + (1 - self._alpha) * self._prev_y

        # Dead zone: skip small deltas
        if self._prev_x is not None:
            dx = abs(sx - self._prev_x)
            dy = abs(sy - self._prev_y)
            if dx < self._dead_zone and dy < self._dead_zone:
                return int(self._prev_x), int(self._prev_y)

        self._prev_x = sx
        self._prev_y = sy

        # Clamp to screen bounds
        out_x = max(0, min(self._screen_w - 1, int(sx)))
        out_y = max(0, min(self._screen_h - 1, int(sy)))
        return out_x, out_y

    def reset(self) -> None:
        """Clear smoothing state (call when gesture switches away)."""
        self._prev_x = None
        self._prev_y = None


class ScrollController:
    """Continuous scroll based on finger Y displacement from anchor.

    When scroll mode starts, the current Y is recorded as anchor.
    Each frame, the delta from anchor determines scroll speed/direction:
      - finger moves UP from anchor → scroll up (positive)
      - finger moves DOWN from anchor → scroll down (negative)
    Speed is proportional to displacement.
    """

    def __init__(self, sensitivity: float = 0.05):
        self._sensitivity = sensitivity
        self._anchor_y: Optional[float] = None

    @property
    def active(self) -> bool:
        return self._anchor_y is not None

    def start_scroll(self, anchor_y: float) -> None:
        """Record the Y position when scroll mode begins."""
        self._anchor_y = anchor_y

    def get_scroll_amount(self, current_y: float) -> int:
        """Compute scroll clicks based on displacement from anchor.

        Args:
            current_y: Current fingertip Y in camera pixel coords.

        Returns:
            Scroll clicks (positive = up, negative = down). 0 if no movement.
        """
        if self._anchor_y is None:
            return 0

        # Delta: anchor - current because camera Y increases downward
        # Moving finger up (smaller Y) → positive delta → scroll up
        delta = self._anchor_y - current_y
        clicks = int(delta * self._sensitivity)
        return clicks

    def stop_scroll(self) -> None:
        """Reset anchor (call when gesture switches away)."""
        self._anchor_y = None
