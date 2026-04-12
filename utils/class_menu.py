import csv

import cv2 as cv
import numpy as np


class ClassMenu:
    """On-screen class selector that reads labels from a CSV file.

    Toggle with Tab, navigate with ↑/↓, confirm with Enter.
    Re-reads CSV each time menu opens (hot-reload).
    """

    def __init__(self, label_csv_path: str, class_counts: dict[int, int] | None = None):
        self._label_csv_path = label_csv_path
        self.labels: list[str] = []
        self.selected_index: int = 0
        self.visible: bool = False
        self._class_counts = class_counts or {}
        self._reload_labels()

    def _reload_labels(self) -> None:
        with open(self._label_csv_path, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            self.labels = [row[0] for row in reader if row]
        # Clamp selected_index
        if self.selected_index >= len(self.labels):
            self.selected_index = max(0, len(self.labels) - 1)

    def set_class_counts(self, counts: dict[int, int]) -> None:
        self._class_counts = counts

    def toggle(self) -> None:
        self.visible = not self.visible
        if self.visible:
            self._reload_labels()

    def move_up(self) -> None:
        if self.visible and self.labels:
            self.selected_index = (self.selected_index - 1) % len(self.labels)

    def move_down(self) -> None:
        if self.visible and self.labels:
            self.selected_index = (self.selected_index + 1) % len(self.labels)

    def confirm(self) -> int | None:
        """Return selected class_id, or None if menu not visible / empty."""
        if not self.visible or not self.labels:
            return None
        class_id = self.selected_index
        self.visible = False
        return class_id

    def draw(self, image: np.ndarray) -> np.ndarray:
        if not self.visible or not self.labels:
            return image

        h, w = image.shape[:2]
        line_h = 32
        pad = 10
        menu_w = 350
        menu_h = line_h * len(self.labels) + pad * 2 + line_h  # +header
        menu_x = 10
        menu_y = 10

        # Semi-transparent background
        overlay = image.copy()
        cv.rectangle(
            overlay,
            (menu_x, menu_y),
            (menu_x + menu_w, menu_y + menu_h),
            (30, 30, 30),
            -1,
        )
        cv.addWeighted(overlay, 0.85, image, 0.15, 0, image)

        # Header
        cv.putText(
            image,
            "SELECT CLASS (Enter to start)",
            (menu_x + pad, menu_y + pad + 20),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
        )

        # Class list
        for i, label in enumerate(self.labels):
            y = menu_y + line_h + pad + i * line_h + 20
            count = self._class_counts.get(i, 0)
            text = f"{i:2d}  {label}"
            count_text = f"{count}"

            if i == self.selected_index:
                # Highlight bar
                cv.rectangle(
                    image,
                    (menu_x + 2, y - 20),
                    (menu_x + menu_w - 2, y + 8),
                    (80, 180, 80),
                    -1,
                )
                color = (255, 255, 255)
            else:
                color = (180, 180, 180)

            cv.putText(
                image, text, (menu_x + pad, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
            cv.putText(
                image,
                count_text,
                (menu_x + menu_w - 60, y),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        return image
