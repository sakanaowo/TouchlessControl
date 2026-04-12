"""Unit tests for utils.class_menu.ClassMenu."""

import os
import tempfile

import numpy as np
import pytest

from utils.class_menu import ClassMenu


@pytest.fixture
def label_file(tmp_path):
    """Create a temporary label CSV with 5 classes."""
    p = tmp_path / "labels.csv"
    p.write_text("null\nopen_palm\nfist\npointer\nthumbs_up\n")
    return str(p)


@pytest.fixture
def menu(label_file):
    return ClassMenu(label_file)


class TestLoadLabels:
    def test_loads_all_labels(self, menu):
        assert menu.labels == ["null", "open_palm", "fist", "pointer", "thumbs_up"]

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("")
        m = ClassMenu(str(p))
        assert m.labels == []

    def test_hot_reload_on_toggle(self, label_file):
        m = ClassMenu(label_file)
        assert len(m.labels) == 5
        # Append a new label
        with open(label_file, "a") as f:
            f.write("new_gesture\n")
        # Labels not updated yet
        assert len(m.labels) == 5
        # Toggle open → triggers reload
        m.toggle()
        assert len(m.labels) == 6
        assert m.labels[-1] == "new_gesture"

    def test_clamps_index_after_label_reduction(self, label_file):
        m = ClassMenu(label_file)
        m.selected_index = 4  # last
        # Overwrite with fewer labels
        with open(label_file, "w") as f:
            f.write("a\nb\n")
        m.toggle()  # open → reload
        assert m.selected_index == 1  # clamped to max


class TestNavigation:
    def test_move_down_wraps(self, menu):
        menu.visible = True
        for _ in range(5):
            menu.move_down()
        assert menu.selected_index == 0  # wrapped

    def test_move_up_wraps(self, menu):
        menu.visible = True
        menu.move_up()
        assert menu.selected_index == 4  # wrapped to last

    def test_navigation_ignored_when_hidden(self, menu):
        assert not menu.visible
        menu.move_down()
        assert menu.selected_index == 0  # unchanged

    def test_sequential_navigation(self, menu):
        menu.visible = True
        menu.move_down()
        menu.move_down()
        assert menu.selected_index == 2
        menu.move_up()
        assert menu.selected_index == 1


class TestConfirm:
    def test_confirm_returns_class_id(self, menu):
        menu.toggle()  # open
        menu.move_down()
        menu.move_down()
        result = menu.confirm()
        assert result == 2
        assert not menu.visible  # auto-close

    def test_confirm_when_hidden_returns_none(self, menu):
        assert menu.confirm() is None

    def test_confirm_empty_labels(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("")
        m = ClassMenu(str(p))
        m.visible = True
        assert m.confirm() is None


class TestToggle:
    def test_toggle_on_off(self, menu):
        assert not menu.visible
        menu.toggle()
        assert menu.visible
        menu.toggle()
        assert not menu.visible


class TestDraw:
    def test_draw_hidden_returns_unchanged(self, menu):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        original = img.copy()
        result = menu.draw(img)
        np.testing.assert_array_equal(result, original)

    def test_draw_visible_modifies_image(self, menu):
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        menu.toggle()
        result = menu.draw(img)
        # Image should be modified (not all zeros)
        assert result.sum() > 0

    def test_draw_with_counts(self, label_file):
        m = ClassMenu(label_file, class_counts={0: 100, 2: 50})
        m.toggle()
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = m.draw(img)
        assert result.sum() > 0


class TestClassCounts:
    def test_set_class_counts(self, menu):
        menu.set_class_counts({0: 10, 1: 20})
        assert menu._class_counts == {0: 10, 1: 20}
