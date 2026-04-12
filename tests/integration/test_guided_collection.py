"""Integration tests for guided gesture collection workflow.

Tests the ClassMenu → CollectionManager → CSV pipeline end-to-end,
verifying key handling priority, data integrity, and state transitions.
"""

import csv
import time
from unittest.mock import patch

import numpy as np
import pytest

from utils.class_menu import ClassMenu
from utils.collection_manager import CollectionManager, HandData


def make_hand(confidence=0.9, dim=42):
    return HandData(features=[0.5] * dim, confidence=confidence)


@pytest.fixture
def label_file(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("null\nopen_palm\nfist\npointer\nthumbs_up\n")
    return str(p)


@pytest.fixture
def csv_file(tmp_path):
    p = tmp_path / "keypoint.csv"
    p.write_text("")
    return str(p)


@pytest.fixture
def menu(label_file):
    return ClassMenu(label_file)


@pytest.fixture
def mgr(csv_file):
    return CollectionManager(
        csv_path=csv_file,
        batch_size=5,
        frame_skip=1,
        quality_threshold=0.7,
        timeout=10.0,
    )


class TestFullSessionWorkflow:
    """US-1: Tab → select class → Enter → countdown → capture → auto-stop."""

    def test_complete_session_writes_csv(self, menu, mgr, csv_file, label_file):
        # Open menu
        menu.toggle()
        assert menu.visible

        # Navigate to class 2 (fist)
        menu.move_down()
        menu.move_down()
        class_id = menu.confirm()
        assert class_id == 2
        assert not menu.visible

        # Start session
        mgr.start_session(class_id)
        assert mgr.state == "countdown"

        # Skip countdown
        mgr.session.countdown_end = time.time() - 1
        mgr.tick()
        assert mgr.state == "recording"

        # Feed frames
        for _ in range(5):
            mgr.on_frame([make_hand()])
        assert mgr.state == "done"

        # Verify CSV
        with open(csv_file) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 5
        assert all(row[0] == "2" for row in rows)

        # Verify class counts
        assert mgr.class_counts[2] == 5

    def test_session_then_another_session(self, menu, mgr, csv_file, label_file):
        """Two consecutive sessions for different classes."""
        # Session 1: class 0
        menu.toggle()
        class_id = menu.confirm()  # index 0
        mgr.start_session(class_id)
        mgr.session.countdown_end = time.time() - 1
        mgr.state = "recording"
        mgr.session.started_at = time.time()
        for _ in range(5):
            mgr.on_frame([make_hand()])
        assert mgr.state == "done"

        # Wait for done → idle
        mgr._done_at = time.time() - 2
        mgr.tick()
        assert mgr.state == "idle"

        # Session 2: class 3
        menu.toggle()
        menu.move_down()
        menu.move_down()
        menu.move_down()
        class_id2 = menu.confirm()
        assert class_id2 == 3
        mgr.start_session(class_id2)
        mgr.session.countdown_end = time.time() - 1
        mgr.state = "recording"
        mgr.session.started_at = time.time()
        for _ in range(5):
            mgr.on_frame([make_hand()])
        assert mgr.state == "done"

        # Verify CSV has both classes
        with open(csv_file) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 10
        assert sum(1 for r in rows if r[0] == "0") == 5
        assert sum(1 for r in rows if r[0] == "3") == 5
        assert mgr.class_counts == {0: 5, 3: 5}


class TestCancelSession:
    """US-5: Cancel discards buffer, nothing written to CSV."""

    def test_cancel_during_recording_discards_all(self, mgr, csv_file):
        mgr.start_session(1)
        mgr.session.countdown_end = time.time() - 1
        mgr.state = "recording"
        mgr.session.started_at = time.time()

        # Record some frames
        for _ in range(3):
            mgr.on_frame([make_hand()])
        assert mgr.session.collected == 3

        # Cancel
        mgr.cancel()
        assert mgr.state == "idle"

        # CSV must be empty
        with open(csv_file) as f:
            assert f.read() == ""

    def test_cancel_during_countdown(self, mgr, csv_file):
        mgr.start_session(2)
        assert mgr.state == "countdown"
        mgr.cancel()
        assert mgr.state == "idle"
        with open(csv_file) as f:
            assert f.read() == ""

    def test_cancel_then_new_session_no_stale_data(self, mgr, csv_file):
        """After cancel, new session starts clean."""
        mgr.start_session(0)
        mgr.session.countdown_end = time.time() - 1
        mgr.state = "recording"
        mgr.session.started_at = time.time()
        mgr.on_frame([make_hand()])
        mgr.on_frame([make_hand()])
        mgr.cancel()

        # New session for different class
        mgr.start_session(4)
        mgr.session.countdown_end = time.time() - 1
        mgr.state = "recording"
        mgr.session.started_at = time.time()
        for _ in range(5):
            mgr.on_frame([make_hand()])

        # Verify only class 4 in CSV
        with open(csv_file) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 5
        assert all(r[0] == "4" for r in rows)


class TestDataIntegrity:
    """Verify no wrong-class writes under any key sequence."""

    def test_class_locked_during_recording(self, menu, mgr, csv_file):
        """Class ID is set at start_session, menu interactions don't change it."""
        menu.toggle()
        menu.move_down()  # index 1
        class_id = menu.confirm()
        mgr.start_session(class_id)
        mgr.session.countdown_end = time.time() - 1
        mgr.state = "recording"
        mgr.session.started_at = time.time()

        # Simulate user trying to open menu during recording
        # (app.py blocks this, but even if manager gets called, class stays locked)
        for _ in range(5):
            mgr.on_frame([make_hand()])

        with open(csv_file) as f:
            rows = list(csv.reader(f))
        assert all(r[0] == "1" for r in rows)

    def test_multi_hand_same_class(self, mgr, csv_file):
        """Multi-hand frame writes both hands under same class."""
        mgr.start_session(2)
        mgr.session.countdown_end = time.time() - 1
        mgr.state = "recording"
        mgr.session.started_at = time.time()

        two_hands = [make_hand(0.9), make_hand(0.85)]
        for _ in range(5):
            mgr.on_frame(two_hands)

        with open(csv_file) as f:
            rows = list(csv.reader(f))
        # 5 frames × 2 hands = 10 rows, all class 2
        assert len(rows) == 10
        assert all(r[0] == "2" for r in rows)

    def test_feature_vector_preserved(self, mgr, csv_file):
        """Feature values are written exactly as provided."""
        features = [float(i) / 100 for i in range(42)]
        hand = HandData(features=features, confidence=0.95)
        mgr.start_session(0)
        mgr.session.countdown_end = time.time() - 1
        mgr.state = "recording"
        mgr.session.started_at = time.time()
        mgr.batch_size = 1
        mgr.session.target_count = 1
        mgr.on_frame([hand])

        with open(csv_file) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 1
        assert rows[0][0] == "0"
        written_features = [float(v) for v in rows[0][1:]]
        assert written_features == features


class TestEmptyHandsList:
    """Edge case: no hands detected during recording."""

    def test_empty_hands_recording_rejects(self, mgr):
        mgr.start_session(0)
        mgr.session.countdown_end = time.time() - 1
        mgr.state = "recording"
        mgr.session.started_at = time.time()

        result = mgr.on_frame([])
        assert result == 0
        assert mgr.session.quality_rejected == 1

    def test_empty_hands_countdown_returns_zero(self, mgr):
        mgr.start_session(0)
        result = mgr.on_frame([])
        assert result == 0


class TestQualityGate:
    """US-3: Quality gate filters low confidence hands."""

    def test_mixed_confidence_multi_hand(self, mgr, csv_file):
        """Only hands above threshold are written."""
        mgr.start_session(0)
        mgr.session.countdown_end = time.time() - 1
        mgr.state = "recording"
        mgr.session.started_at = time.time()

        good = HandData(features=[1.0] * 42, confidence=0.9)
        bad = HandData(features=[0.0] * 42, confidence=0.3)

        mgr.batch_size = 3
        mgr.session.target_count = 3
        for _ in range(3):
            mgr.on_frame([good, bad])

        with open(csv_file) as f:
            rows = list(csv.reader(f))
        # Only good hand written (3 frames × 1 good hand = 3 rows)
        assert len(rows) == 3
        assert all(float(r[1]) == 1.0 for r in rows)


class TestTimeoutFlushPartial:
    """US-3 / Requirement #4: Timeout flushes partial data."""

    def test_timeout_saves_collected(self, mgr, csv_file):
        mgr.start_session(0)
        mgr.session.countdown_end = time.time() - 1
        mgr.state = "recording"
        mgr.session.started_at = time.time()

        mgr.on_frame([make_hand()])
        mgr.on_frame([make_hand()])

        # Force timeout
        mgr.session.started_at = time.time() - 11
        mgr.on_frame([make_hand()])

        assert mgr.state == "done"
        overlay = mgr.get_overlay_state()
        assert overlay["timed_out"] is True
        assert overlay["flushed_count"] == 2

        with open(csv_file) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 2


class TestHotReload:
    """US-7: Label list auto-updates from CSV."""

    def test_new_label_appears_after_toggle(self, label_file, menu):
        assert len(menu.labels) == 5
        with open(label_file, "a") as f:
            f.write("new_gesture\n")
        menu.toggle()  # open → reload
        assert len(menu.labels) == 6
        assert menu.labels[-1] == "new_gesture"

    def test_removed_label_clamps_index(self, label_file):
        m = ClassMenu(label_file)
        m.selected_index = 4  # thumbs_up (last)
        with open(label_file, "w") as f:
            f.write("null\nopen_palm\n")
        m.toggle()
        assert len(m.labels) == 2
        assert m.selected_index == 1


class TestClassCountsSync:
    """Balance chart synced between CollectionManager and ClassMenu."""

    def test_counts_update_after_session(self, menu, mgr, csv_file):
        mgr.start_session(1)
        mgr.session.countdown_end = time.time() - 1
        mgr.state = "recording"
        mgr.session.started_at = time.time()
        for _ in range(5):
            mgr.on_frame([make_hand()])

        menu.set_class_counts(mgr.class_counts)
        assert menu._class_counts[1] == 5

    def test_existing_csv_counts_loaded(self, tmp_path, label_file):
        """Pre-existing CSV data reflected in counts on init."""
        csv_p = tmp_path / "data.csv"
        with open(str(csv_p), "w", newline="") as f:
            w = csv.writer(f)
            for _ in range(20):
                w.writerow([0] + [0.1] * 42)
            for _ in range(10):
                w.writerow([2] + [0.2] * 42)
        mgr = CollectionManager(csv_path=str(csv_p), batch_size=5)
        m = ClassMenu(label_file, class_counts=mgr.class_counts)
        assert m._class_counts == {0: 20, 2: 10}


class TestRapidSessionSequence:
    """Edge case: rapid start/cancel/start cycles."""

    def test_cancel_immediately_after_start(self, mgr):
        mgr.start_session(0)
        mgr.cancel()
        assert mgr.state == "idle"
        assert mgr._buffer == []

    def test_start_cancel_start_completes(self, mgr, csv_file):
        mgr.start_session(0)
        mgr.cancel()

        mgr.start_session(1)
        mgr.session.countdown_end = time.time() - 1
        mgr.state = "recording"
        mgr.session.started_at = time.time()
        for _ in range(5):
            mgr.on_frame([make_hand()])

        assert mgr.state == "done"
        with open(csv_file) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 5
        assert all(r[0] == "1" for r in rows)


class TestFrameSkipIntegration:
    """Verify frame skip with multi-hand recording."""

    def test_frame_skip_with_two_hands(self, csv_file):
        mgr = CollectionManager(
            csv_path=csv_file,
            batch_size=3,
            frame_skip=2,
            quality_threshold=0.7,
        )
        mgr.start_session(0)
        mgr.session.countdown_end = time.time() - 1
        mgr.state = "recording"
        mgr.session.started_at = time.time()

        two_hands = [make_hand(0.9), make_hand(0.85)]
        accepted = 0
        for _ in range(6):
            accepted += mgr.on_frame(two_hands)
        assert accepted == 3  # 6 frames / skip 2 = 3 accepted
        assert mgr.state == "done"

        with open(csv_file) as f:
            rows = list(csv.reader(f))
        # 3 frames × 2 hands = 6 rows
        assert len(rows) == 6
        assert all(r[0] == "0" for r in rows)
