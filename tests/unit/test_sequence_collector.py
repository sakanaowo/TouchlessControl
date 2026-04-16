"""Unit tests for SequenceCollector."""

import numpy as np
import pytest

from utils.sequence_collector import SequenceCollector


@pytest.fixture()
def save_path(tmp_path):
    return str(tmp_path / "test_sequences.npz")


@pytest.fixture()
def collector(save_path):
    return SequenceCollector(
        window_size=4, stride=2, num_features=3, save_path=save_path
    )


class TestRecordingLifecycle:
    def test_initial_state(self, collector):
        assert not collector.is_recording
        assert collector.frames_recorded == 0

    def test_start_sets_recording(self, collector):
        collector.start_recording(class_id=1)
        assert collector.is_recording
        assert collector.frames_recorded == 0

    def test_add_frame_increments(self, collector):
        collector.start_recording(class_id=0)
        collector.add_frame([1.0, 2.0, 3.0])
        collector.add_frame([4.0, 5.0, 6.0])
        assert collector.frames_recorded == 2

    def test_add_frame_noop_when_not_recording(self, collector):
        collector.add_frame([1.0, 2.0, 3.0])
        assert collector.frames_recorded == 0

    def test_cancel_discards_buffer(self, collector):
        collector.start_recording(class_id=2)
        for _ in range(10):
            collector.add_frame([1.0, 2.0, 3.0])
        collector.cancel_recording()
        assert not collector.is_recording
        assert collector.frames_recorded == 0


class TestStopRecording:
    def test_too_short_returns_zero(self, collector):
        """Sessions shorter than window_size produce no sequences."""
        collector.start_recording(class_id=1)
        collector.add_frame([1.0, 2.0, 3.0])
        collector.add_frame([4.0, 5.0, 6.0])
        n = collector.stop_recording()
        assert n == 0

    def test_exact_window_produces_one_sequence(self, collector):
        collector.start_recording(class_id=2)
        for i in range(4):  # window_size = 4
            collector.add_frame([float(i)] * 3)
        n = collector.stop_recording()
        assert n == 1

    def test_sliding_window_count(self, collector):
        """With 8 frames, window=4, stride=2 → sequences at 0,2,4 → 3 seqs."""
        collector.start_recording(class_id=0)
        for i in range(8):
            collector.add_frame([float(i)] * 3)
        n = collector.stop_recording()
        assert n == 3

    def test_npz_content_correct(self, collector, save_path):
        collector.start_recording(class_id=1)
        for i in range(6):  # 6 frames, window=4, stride=2 → seq at 0,2 → 2 seqs
            collector.add_frame([float(i)] * 3)
        n = collector.stop_recording()
        assert n == 2

        data = np.load(save_path)
        assert data["sequences"].shape == (2, 4, 3)
        assert data["labels"].shape == (2,)
        np.testing.assert_array_equal(data["labels"], [1, 1])

        # First window = frames 0,1,2,3
        np.testing.assert_array_equal(data["sequences"][0, :, 0], [0.0, 1.0, 2.0, 3.0])
        # Second window = frames 2,3,4,5
        np.testing.assert_array_equal(data["sequences"][1, :, 0], [2.0, 3.0, 4.0, 5.0])

    def test_stop_when_not_recording_returns_zero(self, collector):
        assert collector.stop_recording() == 0


class TestAppendMode:
    def test_multiple_sessions_accumulate(self, collector, save_path):
        # Session 1
        collector.start_recording(class_id=0)
        for i in range(4):
            collector.add_frame([float(i)] * 3)
        n1 = collector.stop_recording()
        assert n1 == 1

        # Session 2
        collector.start_recording(class_id=2)
        for i in range(6):
            collector.add_frame([float(i + 10)] * 3)
        n2 = collector.stop_recording()
        assert n2 == 2

        data = np.load(save_path)
        assert data["sequences"].shape[0] == 3  # 1 + 2
        np.testing.assert_array_equal(data["labels"], [0, 2, 2])


class TestLoadExisting:
    def test_empty_when_no_file(self, collector):
        assert collector.load_existing() == {}

    def test_counts_by_class(self, collector, save_path):
        # Create data with known labels
        collector.start_recording(class_id=0)
        for _ in range(4):
            collector.add_frame([1.0] * 3)
        collector.stop_recording()

        collector.start_recording(class_id=2)
        for _ in range(6):
            collector.add_frame([2.0] * 3)
        collector.stop_recording()

        counts = collector.load_existing()
        assert counts[0] == 1
        assert counts[2] == 2


class TestEdgeCases:
    def test_stride_one(self, save_path):
        """Stride=1 maximizes overlap."""
        c = SequenceCollector(
            window_size=3, stride=1, num_features=2, save_path=save_path
        )
        c.start_recording(class_id=0)
        for i in range(5):  # 5 frames, window=3, stride=1 → 3 sequences
            c.add_frame([float(i)] * 2)
        n = c.stop_recording()
        assert n == 3

    def test_numpy_input(self, collector):
        collector.start_recording(class_id=0)
        collector.add_frame(np.array([1.0, 2.0, 3.0]))
        assert collector.frames_recorded == 1
