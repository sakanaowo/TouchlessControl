"""Unit tests for SequenceBuffer."""

import numpy as np
import pytest

from utils.sequence_buffer import SequenceBuffer


class TestSequenceBufferInit:
    def test_defaults(self):
        buf = SequenceBuffer()
        assert buf.window_size == 20
        assert buf.num_features == 93
        assert buf.count == 0
        assert not buf.is_ready()

    def test_custom_size(self):
        buf = SequenceBuffer(window_size=5, num_features=10)
        assert buf.window_size == 5
        assert buf.num_features == 10

    def test_initial_window_is_zeros(self):
        buf = SequenceBuffer(window_size=3, num_features=2)
        w = buf.get_window()
        assert w.shape == (3, 2)
        np.testing.assert_array_equal(w, 0.0)


class TestPushAndGetWindow:
    def test_single_push_partial_fill(self):
        buf = SequenceBuffer(window_size=3, num_features=2)
        buf.push([1.0, 2.0])
        w = buf.get_window()
        assert w.shape == (3, 2)
        # zero-padded at front, data at last row
        np.testing.assert_array_equal(w[0], [0.0, 0.0])
        np.testing.assert_array_equal(w[1], [0.0, 0.0])
        np.testing.assert_array_equal(w[2], [1.0, 2.0])

    def test_two_pushes_partial_fill(self):
        buf = SequenceBuffer(window_size=3, num_features=2)
        buf.push([1.0, 2.0])
        buf.push([3.0, 4.0])
        w = buf.get_window()
        np.testing.assert_array_equal(w[0], [0.0, 0.0])
        np.testing.assert_array_equal(w[1], [1.0, 2.0])
        np.testing.assert_array_equal(w[2], [3.0, 4.0])

    def test_exact_fill(self):
        buf = SequenceBuffer(window_size=3, num_features=2)
        for i in range(3):
            buf.push([float(i), float(i + 10)])
        assert buf.is_ready()
        w = buf.get_window()
        np.testing.assert_array_equal(w[0], [0.0, 10.0])
        np.testing.assert_array_equal(w[1], [1.0, 11.0])
        np.testing.assert_array_equal(w[2], [2.0, 12.0])

    def test_circular_overwrite(self):
        buf = SequenceBuffer(window_size=3, num_features=1)
        for i in range(5):  # push 0,1,2,3,4 → window = [2,3,4]
            buf.push([float(i)])
        assert buf.count == 5
        w = buf.get_window()
        np.testing.assert_array_equal(w.flatten(), [2.0, 3.0, 4.0])

    def test_overwrite_preserves_chronological_order(self):
        buf = SequenceBuffer(window_size=4, num_features=1)
        for i in range(7):  # push 0..6 → window = [3,4,5,6]
            buf.push([float(i)])
        w = buf.get_window()
        np.testing.assert_array_equal(w.flatten(), [3.0, 4.0, 5.0, 6.0])

    def test_exact_multiple_of_window(self):
        """When count == k * window_size, start index is 0."""
        buf = SequenceBuffer(window_size=3, num_features=1)
        for i in range(6):  # push 0..5 → window = [3,4,5]
            buf.push([float(i)])
        w = buf.get_window()
        np.testing.assert_array_equal(w.flatten(), [3.0, 4.0, 5.0])


class TestGetWindowReturnsNewArray:
    def test_mutation_does_not_affect_buffer(self):
        buf = SequenceBuffer(window_size=2, num_features=1)
        buf.push([1.0])
        buf.push([2.0])
        w = buf.get_window()
        w[0, 0] = 999.0
        w2 = buf.get_window()
        np.testing.assert_array_equal(w2.flatten(), [1.0, 2.0])


class TestClear:
    def test_clear_resets_state(self):
        buf = SequenceBuffer(window_size=3, num_features=2)
        for i in range(5):
            buf.push([float(i)] * 2)
        assert buf.is_ready()
        buf.clear()
        assert buf.count == 0
        assert not buf.is_ready()
        np.testing.assert_array_equal(buf.get_window(), 0.0)

    def test_push_after_clear(self):
        buf = SequenceBuffer(window_size=2, num_features=1)
        buf.push([10.0])
        buf.push([20.0])
        buf.clear()
        buf.push([30.0])
        w = buf.get_window()
        np.testing.assert_array_equal(w.flatten(), [0.0, 30.0])


class TestIsReady:
    def test_not_ready_until_full(self):
        buf = SequenceBuffer(window_size=3, num_features=1)
        assert not buf.is_ready()
        buf.push([1.0])
        assert not buf.is_ready()
        buf.push([2.0])
        assert not buf.is_ready()
        buf.push([3.0])
        assert buf.is_ready()

    def test_stays_ready_after_more_pushes(self):
        buf = SequenceBuffer(window_size=2, num_features=1)
        buf.push([1.0])
        buf.push([2.0])
        buf.push([3.0])
        assert buf.is_ready()


class TestEdgeCases:
    def test_window_size_one(self):
        buf = SequenceBuffer(window_size=1, num_features=3)
        buf.push([1.0, 2.0, 3.0])
        w = buf.get_window()
        np.testing.assert_array_equal(w, [[1.0, 2.0, 3.0]])

    def test_numpy_input(self):
        buf = SequenceBuffer(window_size=2, num_features=2)
        buf.push(np.array([1.0, 2.0]))
        buf.push(np.array([3.0, 4.0]))
        w = buf.get_window()
        np.testing.assert_array_equal(w[0], [1.0, 2.0])
        np.testing.assert_array_equal(w[1], [3.0, 4.0])

    def test_dtype_is_float32(self):
        buf = SequenceBuffer(window_size=2, num_features=1)
        buf.push([1])  # int input
        w = buf.get_window()
        assert w.dtype == np.float32
