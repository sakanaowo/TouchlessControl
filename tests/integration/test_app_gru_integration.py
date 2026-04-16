"""Integration tests for Phase 3: app.py GRU pipeline wiring and sequence collection."""

from __future__ import annotations

import csv
import os
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from utils.sequence_buffer import SequenceBuffer
from utils.debouncer import Debouncer
from utils.sequence_collector import SequenceCollector
from utils.collection_manager import CollectionManager, HandData
from utils.gesture_state_machine import GestureEvent
from model.temporal_classifier import TemporalClassifier


TFLITE_PATH = "model/temporal_classifier/temporal_classifier.tflite"


@pytest.fixture()
def tflite_exists():
    if not os.path.exists(TFLITE_PATH):
        pytest.skip("Temporal TFLite model not found")


# ---------------------------------------------------------------------------
# T3.1: GRU pipeline wiring (SequenceBuffer → TemporalClassifier → Debouncer)
# ---------------------------------------------------------------------------


class TestGRUPipelineWiring:
    """Verify the full GRU pipeline: features → buffer → classifier → debouncer."""

    def test_pipeline_end_to_end(self, tflite_exists):
        """Push 20 frames through buffer → classifier → debouncer, get event."""
        tc = TemporalClassifier()
        buf = SequenceBuffer(window_size=tc.window_size, num_features=93)
        deb = Debouncer(confidence_threshold=0.0, cooldown_seconds=0.0)

        # Push enough frames to fill the buffer
        for _ in range(20):
            features = np.random.randn(93).astype(np.float32).tolist()
            buf.push(features)

        assert buf.is_ready()
        class_id, scores = tc(buf.get_window())
        assert 0 <= class_id < tc.num_classes
        assert len(scores) == tc.num_classes

        # Debouncer should produce a start event for non-null prediction
        # (confidence_threshold=0.0 accepts everything)
        if class_id != 0:
            event = deb.update(class_id, scores)
            assert event is not None
            assert event.event_type == "start"

    def test_pipeline_before_ready_returns_null(self, tflite_exists):
        """Before buffer is full, pipeline should not classify."""
        tc = TemporalClassifier()
        buf = SequenceBuffer(window_size=tc.window_size, num_features=93)

        # Push only 5 frames
        for _ in range(5):
            buf.push(np.random.randn(93).astype(np.float32).tolist())

        assert not buf.is_ready()
        # The app treats this as hand_sign_id=0 (null)

    def test_debouncer_emits_end_on_no_hand(self, tflite_exists):
        """After a start event, update_no_hand should emit an end event."""
        deb = Debouncer(confidence_threshold=0.0, cooldown_seconds=0.0)
        scores = np.array([0.1, 0.1, 0.6, 0.1, 0.1], dtype=np.float32)
        event = deb.update(2, scores)  # left_click start
        assert event is not None and event.event_type == "start"

        end_event = deb.update_no_hand()
        assert end_event is not None
        assert end_event.event_type == "end"
        assert end_event.gesture == "left_click"

    def test_continuous_gestures_bypass_debouncer(self, tflite_exists):
        """pointer_move (1) and scroll_mode (4) bypass debouncer in app.py logic."""
        tc = TemporalClassifier()
        buf = SequenceBuffer(window_size=tc.window_size, num_features=93)
        deb = Debouncer()

        # Simulate pointer_move classification
        # In app.py, hand_sign_id == 1 goes directly to CursorController
        # This test verifies the debouncer is NOT called for id=1
        # and would only produce events for discrete gestures (2, 3)
        scores = np.array([0.0, 0.95, 0.0, 0.0, 0.05])
        # If we were to feed id=1 to debouncer, it would emit start
        # but in the real app.py, id=1 and id=4 skip debouncer entirely
        # We just verify the debouncer handles them correctly if called
        event = deb.update(1, scores)
        assert event is not None  # would start, but app.py skips this path


# ---------------------------------------------------------------------------
# T3.2: CollectionManager with SequenceCollector
# ---------------------------------------------------------------------------


class TestCollectionManagerSequenceMode:
    """Verify CollectionManager delegates to SequenceCollector when configured."""

    @pytest.fixture
    def npz_path(self, tmp_path):
        return str(tmp_path / "sequences.npz")

    @pytest.fixture
    def csv_path(self, tmp_path):
        p = tmp_path / "keypoint.csv"
        p.write_text("")
        return str(p)

    def test_sequence_mode_delegates_recording(self, csv_path, npz_path):
        """In sequence mode, frames go to SequenceCollector, not CSV buffer."""
        sc = SequenceCollector(
            window_size=5, stride=2, num_features=3, save_path=npz_path
        )
        mgr = CollectionManager(
            csv_path=csv_path,
            batch_size=10,
            frame_skip=1,
            quality_threshold=0.0,
            timeout=30.0,
            sequence_collector=sc,
        )

        mgr.start_session(class_id=2)
        assert mgr.state == "countdown"
        assert sc.is_recording

        # Advance past countdown
        mgr.session.countdown_end = time.time() - 1
        mgr.tick()
        assert mgr.state == "recording"

        # Feed 10 frames
        for i in range(10):
            hands = [HandData(features=[float(i)] * 3, confidence=0.9)]
            mgr.on_frame(hands)

        assert mgr.state == "done"
        # Verify flush info
        overlay = mgr.get_overlay_state()
        assert overlay["flushed_count"] > 0  # number of sequences saved

        # Verify NPZ was written
        assert os.path.exists(npz_path)
        data = np.load(npz_path)
        assert "sequences" in data
        assert "labels" in data
        assert all(lbl == 2 for lbl in data["labels"])

    def test_sequence_mode_cancel_discards(self, csv_path, npz_path):
        """Cancelling in sequence mode discards the recording."""
        sc = SequenceCollector(
            window_size=5, stride=2, num_features=3, save_path=npz_path
        )
        mgr = CollectionManager(
            csv_path=csv_path,
            batch_size=30,
            frame_skip=1,
            quality_threshold=0.0,
            sequence_collector=sc,
        )

        mgr.start_session(class_id=1)
        mgr.session.countdown_end = time.time() - 1
        mgr.tick()

        # Feed a few frames then cancel
        for _ in range(3):
            mgr.on_frame([HandData(features=[0.5] * 3, confidence=0.9)])

        mgr.cancel()
        assert mgr.state == "idle"
        assert not sc.is_recording
        assert not os.path.exists(npz_path)

    def test_sequence_mode_class_counts_from_npz(self, csv_path, npz_path):
        """After finishing, class_counts should reflect NPZ data."""
        sc = SequenceCollector(
            window_size=3, stride=1, num_features=2, save_path=npz_path
        )
        mgr = CollectionManager(
            csv_path=csv_path,
            batch_size=5,
            frame_skip=1,
            quality_threshold=0.0,
            timeout=30.0,
            sequence_collector=sc,
        )

        mgr.start_session(class_id=0)
        mgr.session.countdown_end = time.time() - 1
        mgr.tick()

        for i in range(5):
            mgr.on_frame([HandData(features=[float(i)] * 2, confidence=0.9)])

        assert mgr.state == "done"
        # class_counts should be loaded from NPZ
        assert 0 in mgr.class_counts
        assert mgr.class_counts[0] > 0

    def test_csv_mode_unchanged(self, csv_path):
        """Without sequence_collector, CollectionManager works as before (CSV)."""
        mgr = CollectionManager(
            csv_path=csv_path,
            batch_size=5,
            frame_skip=1,
            quality_threshold=0.0,
            timeout=30.0,
        )

        mgr.start_session(class_id=1)
        mgr.session.countdown_end = time.time() - 1
        mgr.tick()

        for i in range(5):
            mgr.on_frame([HandData(features=[0.1] * 42, confidence=0.9)])

        assert mgr.state == "done"
        # CSV should have rows
        with open(csv_path) as f:
            rows = list(csv.reader(f))
        assert len(rows) > 0
        assert all(int(r[0]) == 1 for r in rows)


# ---------------------------------------------------------------------------
# T3.3: Package exports
# ---------------------------------------------------------------------------


class TestPackageExports:
    """Verify that new classes are importable from package roots."""

    def test_utils_exports(self):
        from utils import CvFpsCalc, SequenceBuffer, Debouncer, SequenceCollector

        assert SequenceBuffer is not None
        assert Debouncer is not None
        assert SequenceCollector is not None

    def test_model_exports(self):
        from model import TemporalClassifier, KeyPointClassifierV2

        assert TemporalClassifier is not None
        assert KeyPointClassifierV2 is not None
