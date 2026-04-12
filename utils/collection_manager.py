"""Session-based gesture data collection with RAM buffer and quality gate."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from typing import NamedTuple


class HandData(NamedTuple):
    """Per-hand data passed into CollectionManager.on_frame()."""

    features: list[float]  # pre-processed feature vector (any dim)
    confidence: float  # hand detection confidence [0, 1]


@dataclass
class CollectionSession:
    class_id: int
    target_count: int = 30
    collected: int = 0  # frames accepted (not rows)
    countdown_end: float = 0.0
    quality_rejected: int = 0
    started_at: float = 0.0
    timeout: float = 10.0


class CollectionManager:
    """FSM: idle → countdown → recording → done → idle.

    RAM buffer collects feature vectors and flushes to CSV only on success.
    Cancel discards the entire buffer.
    """

    COUNTDOWN_SECONDS = 3
    DONE_DISPLAY_SECONDS = 1.0

    def __init__(
        self,
        csv_path: str = "model/keypoint_classifier/keypoint.csv",
        batch_size: int = 30,
        frame_skip: int = 2,
        quality_threshold: float = 0.7,
        timeout: float = 10.0,
    ):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.frame_skip = frame_skip
        self.quality_threshold = quality_threshold
        self.timeout = timeout

        self.state: str = "idle"  # idle | countdown | recording | done
        self.session: CollectionSession | None = None
        self.class_counts: dict[int, int] = {}
        self._buffer: list[list[float]] = []
        self._frame_counter: int = 0
        self._done_at: float = 0.0
        self._last_flush_count: int = 0
        self._last_flush_target: int = 0
        self._last_flush_timed_out: bool = False
        self._last_flush_class_id: int = -1

        self._load_existing_counts()

    # ── Public API ──────────────────────────────────────────────

    def start_session(self, class_id: int) -> None:
        """Start a collection session for *class_id* (from ClassMenu.confirm())."""
        self._buffer.clear()
        self._frame_counter = 0
        self.session = CollectionSession(
            class_id=class_id,
            target_count=self.batch_size,
            countdown_end=time.time() + self.COUNTDOWN_SECONDS,
            started_at=time.time(),
            timeout=self.timeout,
        )
        self.state = "countdown"

    def on_frame(self, hands: list[HandData]) -> int:
        """Process one video frame.  Returns 0 or 1 (frame accepted?).

        Multi-hand: if frame passes skip + quality, each qualifying hand
        adds a row to the buffer, but ``collected`` increments by 1 (frame-based).
        """
        if self.state == "countdown":
            if time.time() >= self.session.countdown_end:
                self.state = "recording"
                self.session.started_at = time.time()
            return 0

        if self.state != "recording" or self.session is None:
            return 0

        # Timeout check
        elapsed = time.time() - self.session.started_at
        if elapsed >= self.session.timeout:
            self._finish(timed_out=True)
            return 0

        # Frame skip
        self._frame_counter += 1
        if self._frame_counter % self.frame_skip != 0:
            return 0

        # Quality gate: at least one hand must pass
        good_hands = [h for h in hands if h.confidence >= self.quality_threshold]
        if not good_hands:
            self.session.quality_rejected += 1
            return 0

        # Accept frame — add each qualifying hand as a buffer row
        for h in good_hands:
            self._buffer.append([self.session.class_id, *h.features])

        self.session.collected += 1

        # Check if target reached
        if self.session.collected >= self.session.target_count:
            self._finish(timed_out=False)

        return 1

    def cancel(self) -> None:
        """Discard entire buffer, return to idle."""
        self._buffer.clear()
        self.session = None
        self.state = "idle"

    def tick(self) -> None:
        """Call each frame to handle auto-transitions (done → idle)."""
        if self.state == "countdown" and self.session is not None:
            if time.time() >= self.session.countdown_end:
                self.state = "recording"
                self.session.started_at = time.time()

        if self.state == "done":
            if time.time() - self._done_at >= self.DONE_DISPLAY_SECONDS:
                self.state = "idle"

    def get_overlay_state(self) -> dict:
        """Return current state info for overlay rendering."""
        base = {"state": self.state}
        if self.session is not None:
            base["class_id"] = self.session.class_id
            base["collected"] = self.session.collected
            base["target"] = self.session.target_count
            base["quality_rejected"] = self.session.quality_rejected
            if self.state == "countdown":
                remaining = max(0, self.session.countdown_end - time.time())
                base["countdown_remaining"] = remaining
        if self.state == "done":
            base["flushed_count"] = self._last_flush_count
            base["flushed_target"] = self._last_flush_target
            base["timed_out"] = self._last_flush_timed_out
            base["class_id"] = self._last_flush_class_id
        return base

    def adjust_batch_size(self, delta: int) -> None:
        """Adjust batch size by *delta* (clamped to [5, 200])."""
        self.batch_size = max(5, min(200, self.batch_size + delta))

    # ── Internal ────────────────────────────────────────────────

    def _finish(self, *, timed_out: bool) -> None:
        """Flush buffer to CSV and transition to done."""
        self._flush_to_csv()
        self._last_flush_count = self.session.collected
        self._last_flush_target = self.session.target_count
        self._last_flush_timed_out = timed_out
        self._last_flush_class_id = self.session.class_id
        # Update class counts
        self.class_counts[self.session.class_id] = self.class_counts.get(
            self.session.class_id, 0
        ) + len(self._buffer)
        self._buffer.clear()
        self.session = None
        self._done_at = time.time()
        self.state = "done"

    def _flush_to_csv(self) -> None:
        """Append buffered rows to CSV file."""
        if not self._buffer:
            return
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for row in self._buffer:
                writer.writerow(row)

    def _load_existing_counts(self) -> None:
        """Count existing samples per class from CSV."""
        self.class_counts.clear()
        try:
            with open(self.csv_path, newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        try:
                            cid = int(row[0])
                            self.class_counts[cid] = self.class_counts.get(cid, 0) + 1
                        except (ValueError, IndexError):
                            continue
        except FileNotFoundError:
            pass
