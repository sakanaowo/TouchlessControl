"""Unit tests for utils.collection_manager.CollectionManager."""

import csv
import time
from unittest.mock import patch

import pytest

from utils.collection_manager import CollectionManager, CollectionSession, HandData


@pytest.fixture
def csv_file(tmp_path):
    """Empty CSV file for test output."""
    p = tmp_path / "keypoint.csv"
    p.write_text("")
    return str(p)


@pytest.fixture
def csv_with_data(tmp_path):
    """CSV with pre-existing samples."""
    p = tmp_path / "keypoint.csv"
    with open(str(p), "w", newline="") as f:
        w = csv.writer(f)
        for _ in range(10):
            w.writerow([0] + [0.1] * 42)
        for _ in range(5):
            w.writerow([1] + [0.2] * 42)
    return str(p)


@pytest.fixture
def mgr(csv_file):
    return CollectionManager(
        csv_path=csv_file,
        batch_size=5,
        frame_skip=1,
        quality_threshold=0.7,
        timeout=10.0,
    )


def make_hand(confidence=0.9, dim=42):
    """Create a HandData with dummy features."""
    return HandData(features=[0.5] * dim, confidence=confidence)


class TestInit:
    def test_initial_state(self, mgr):
        assert mgr.state == "idle"
        assert mgr.session is None
        assert mgr.class_counts == {}

    def test_loads_existing_counts(self, csv_with_data):
        m = CollectionManager(csv_path=csv_with_data, batch_size=5)
        assert m.class_counts == {0: 10, 1: 5}

    def test_missing_csv_no_error(self, tmp_path):
        m = CollectionManager(csv_path=str(tmp_path / "nonexistent.csv"))
        assert m.class_counts == {}


class TestStartSession:
    def test_transitions_to_countdown(self, mgr):
        mgr.start_session(3)
        assert mgr.state == "countdown"
        assert mgr.session is not None
        assert mgr.session.class_id == 3
        assert mgr.session.target_count == 5

    def test_clears_previous_buffer(self, mgr):
        mgr._buffer = [[1, 2, 3]]
        mgr.start_session(0)
        assert mgr._buffer == []


class TestCountdown:
    def test_on_frame_during_countdown_returns_zero(self, mgr):
        mgr.start_session(0)
        assert mgr.state == "countdown"
        result = mgr.on_frame([make_hand()])
        assert result == 0

    def test_transitions_to_recording_after_countdown(self, mgr):
        mgr.start_session(0)
        # Force countdown to be in the past
        mgr.session.countdown_end = time.time() - 1
        mgr.on_frame([make_hand()])
        assert mgr.state == "recording"

    def test_tick_transitions_countdown_to_recording(self, mgr):
        mgr.start_session(0)
        mgr.session.countdown_end = time.time() - 1
        mgr.tick()
        assert mgr.state == "recording"


class TestRecording:
    def _start_recording(self, mgr):
        mgr.start_session(0)
        mgr.session.countdown_end = time.time() - 1
        mgr.state = "recording"
        mgr.session.started_at = time.time()

    def test_accepts_good_hand(self, mgr):
        self._start_recording(mgr)
        result = mgr.on_frame([make_hand(confidence=0.9)])
        assert result == 1
        assert mgr.session.collected == 1
        assert len(mgr._buffer) == 1

    def test_rejects_low_confidence(self, mgr):
        self._start_recording(mgr)
        result = mgr.on_frame([make_hand(confidence=0.3)])
        assert result == 0
        assert mgr.session.collected == 0
        assert mgr.session.quality_rejected == 1

    def test_multi_hand_counts_one_frame(self, mgr):
        self._start_recording(mgr)
        hands = [make_hand(0.9), make_hand(0.8)]
        result = mgr.on_frame(hands)
        assert result == 1
        assert mgr.session.collected == 1
        # But 2 rows in buffer
        assert len(mgr._buffer) == 2

    def test_multi_hand_filters_low_confidence(self, mgr):
        self._start_recording(mgr)
        hands = [make_hand(0.9), make_hand(0.3)]
        result = mgr.on_frame(hands)
        assert result == 1
        assert len(mgr._buffer) == 1  # only the good hand

    def test_all_hands_low_confidence_rejected(self, mgr):
        self._start_recording(mgr)
        hands = [make_hand(0.5), make_hand(0.3)]
        result = mgr.on_frame(hands)
        assert result == 0
        assert mgr.session.quality_rejected == 1

    def test_auto_stop_at_target(self, mgr):
        self._start_recording(mgr)
        for _ in range(5):
            mgr.on_frame([make_hand()])
        assert mgr.state == "done"

    def test_buffer_flushed_to_csv(self, mgr, csv_file):
        self._start_recording(mgr)
        for _ in range(5):
            mgr.on_frame([make_hand()])
        # Verify CSV has data
        with open(csv_file) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 5
        assert rows[0][0] == "0"  # class_id

    def test_class_counts_updated_after_flush(self, mgr):
        self._start_recording(mgr)
        for _ in range(5):
            mgr.on_frame([make_hand()])
        assert mgr.class_counts[0] == 5


class TestFrameSkip:
    def test_skips_frames(self, csv_file):
        m = CollectionManager(csv_path=csv_file, batch_size=5, frame_skip=3)
        m.start_session(0)
        m.session.countdown_end = time.time() - 1
        m.state = "recording"
        m.session.started_at = time.time()

        accepted = 0
        for _ in range(15):
            accepted += m.on_frame([make_hand()])
        assert accepted == 5  # 15 frames / skip 3 = 5
        assert m.state == "done"

    def test_frame_skip_2(self, csv_file):
        m = CollectionManager(csv_path=csv_file, batch_size=3, frame_skip=2)
        m.start_session(0)
        m.session.countdown_end = time.time() - 1
        m.state = "recording"
        m.session.started_at = time.time()

        accepted = 0
        for _ in range(6):
            accepted += m.on_frame([make_hand()])
        assert accepted == 3


class TestTimeout:
    def test_timeout_flushes_partial(self, mgr, csv_file):
        mgr.start_session(0)
        mgr.session.countdown_end = time.time() - 1
        mgr.state = "recording"
        mgr.session.started_at = time.time() - 11  # already timed out

        # Collect 2 of 5 first
        mgr.session.started_at = time.time()
        mgr.on_frame([make_hand()])
        mgr.on_frame([make_hand()])
        assert mgr.session.collected == 2

        # Now trigger timeout
        mgr.session.started_at = time.time() - 11
        mgr.on_frame([make_hand()])

        assert mgr.state == "done"
        overlay = mgr.get_overlay_state()
        assert overlay["timed_out"] is True
        assert overlay["flushed_count"] == 2

        # CSV should have 2 rows
        with open(csv_file) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 2


class TestCancel:
    def test_cancel_discards_buffer(self, mgr, csv_file):
        mgr.start_session(0)
        mgr.session.countdown_end = time.time() - 1
        mgr.state = "recording"
        mgr.session.started_at = time.time()

        mgr.on_frame([make_hand()])
        mgr.on_frame([make_hand()])
        assert len(mgr._buffer) == 2

        mgr.cancel()
        assert mgr.state == "idle"
        assert mgr._buffer == []
        assert mgr.session is None

        # CSV should be empty
        with open(csv_file) as f:
            assert f.read() == ""

    def test_cancel_during_countdown(self, mgr):
        mgr.start_session(0)
        assert mgr.state == "countdown"
        mgr.cancel()
        assert mgr.state == "idle"


class TestDoneTransition:
    def test_done_auto_returns_to_idle(self, mgr):
        mgr.state = "done"
        mgr._done_at = time.time() - 2  # past display time
        mgr.tick()
        assert mgr.state == "idle"

    def test_done_stays_during_display(self, mgr):
        mgr.state = "done"
        mgr._done_at = time.time()  # just now
        mgr.tick()
        assert mgr.state == "done"


class TestOverlayState:
    def test_idle_state(self, mgr):
        info = mgr.get_overlay_state()
        assert info["state"] == "idle"

    def test_countdown_state(self, mgr):
        mgr.start_session(2)
        info = mgr.get_overlay_state()
        assert info["state"] == "countdown"
        assert info["class_id"] == 2
        assert "countdown_remaining" in info

    def test_done_state(self, mgr):
        mgr.state = "done"
        mgr._last_flush_count = 5
        mgr._last_flush_target = 5
        mgr._last_flush_timed_out = False
        mgr._last_flush_class_id = 2
        info = mgr.get_overlay_state()
        assert info["state"] == "done"
        assert info["flushed_count"] == 5
        assert info["timed_out"] is False
        assert info["class_id"] == 2


class TestBatchSize:
    def test_adjust_batch_size(self, mgr):
        mgr.adjust_batch_size(10)
        assert mgr.batch_size == 15

    def test_clamp_min(self, mgr):
        mgr.adjust_batch_size(-100)
        assert mgr.batch_size == 5

    def test_clamp_max(self, mgr):
        mgr.adjust_batch_size(300)
        assert mgr.batch_size == 200


class TestOnFrameIdleState:
    def test_on_frame_when_idle_returns_zero(self, mgr):
        result = mgr.on_frame([make_hand()])
        assert result == 0

    def test_on_frame_when_done_returns_zero(self, mgr):
        mgr.state = "done"
        result = mgr.on_frame([make_hand()])
        assert result == 0
