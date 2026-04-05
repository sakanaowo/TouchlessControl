import os
import shutil
import subprocess
import time

import yaml

from utils.gesture_state_machine import GestureEvent

_YAML_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "gesture_actions.yaml"
)


class ActionMapper:
    """
    Translate GestureEvents to OS input actions.

    Backend selection at init time (deterministic, no runtime switching):
      1. If DISPLAY is set (X11 / XWayland) → pynput (fast, no extra install).
      2. Else if WAYLAND_DISPLAY is set AND ydotool is available → ydotool subprocess.
      3. Otherwise → raise RuntimeError so the caller can add --no-actions flag.
    """

    def __init__(self, config_path: str = _YAML_PATH):
        with open(config_path, "r") as f:
            self._config: dict = yaml.safe_load(f)

        self._backend = self._detect_backend()
        self._dragging: bool = False

        if self._backend == "pynput":
            from pynput.keyboard import Controller as KbCtrl, Key as PynKey
            from pynput.mouse import Controller as MsCtrl, Button as PynBtn

            self._kb = KbCtrl()
            self._ms = MsCtrl()
            self._key_cls = PynKey
            self._btn_cls = PynBtn

    def _detect_backend(self) -> str:
        if os.environ.get("DISPLAY"):
            return "pynput"
        if os.environ.get("WAYLAND_DISPLAY"):
            if shutil.which("ydotool"):
                return "ydotool"
            raise RuntimeError(
                "Wayland session detected but neither DISPLAY (XWayland) nor ydotool "
                "is available. Install ydotool ('sudo apt install ydotool') or start "
                "app with --no-actions."
            )
        raise RuntimeError(
            "No display backend found (no DISPLAY, no WAYLAND_DISPLAY). "
            "Run with --no-actions flag."
        )

    def handle(self, event: GestureEvent) -> None:
        """
        Dispatch an OS action for the given GestureEvent.
        'start' always fires. 'hold' fires only when repeat: true in config.
        'end' fires for mouse_drag gestures to release the button.
        """
        action = self._config.get(event.gesture)
        if action is None:
            return

        on_start = action.get("on_start")

        # mouse_drag: press on start, release on end
        if on_start == "mouse_drag":
            if event.event_type == "start":
                self._mouse_press(action["button"])
                self._dragging = True
            elif event.event_type == "end" and self._dragging:
                self._mouse_release(action["button"])
                self._dragging = False
            return

        if event.event_type == "end":
            return

        if event.event_type == "hold" and not action.get("repeat", False):
            return

        if on_start == "key_press":
            self._key_press(action["key"])
        elif on_start == "key_combo":
            self._key_combo(action["keys"])
        elif on_start == "mouse_click":
            self._mouse_click(action["button"])
        elif on_start == "scroll":
            self._scroll(action.get("direction", "down"), action.get("amount", 3))

    def _key_press(self, key_name: str) -> None:
        if self._backend == "pynput":
            try:
                k = getattr(self._key_cls, key_name, None) or key_name
                self._kb.press(k)
                self._kb.release(k)
            except Exception:
                pass
        elif self._backend == "ydotool":
            subprocess.run(["ydotool", "key", key_name], check=False)

    def _key_combo(self, key_names: list) -> None:
        if self._backend == "pynput":
            keys = [getattr(self._key_cls, k, k) for k in key_names]
            try:
                for k in keys:
                    self._kb.press(k)
                for k in reversed(keys):
                    self._kb.release(k)
            except Exception:
                pass
        elif self._backend == "ydotool":
            combo = "+".join(key_names)
            subprocess.run(["ydotool", "key", combo], check=False)

    def _mouse_click(self, button_name: str) -> None:
        if self._backend == "pynput":
            try:
                button = getattr(self._btn_cls, button_name)
                self._ms.click(button)
            except Exception:
                pass
        elif self._backend == "ydotool":
            btn_map = {"left": "0x40001", "right": "0x40002", "middle": "0x40003"}
            code = btn_map.get(button_name, "0x40001")
            subprocess.run(["ydotool", "click", code], check=False)

    def _mouse_press(self, button_name: str) -> None:
        if self._backend == "pynput":
            try:
                button = getattr(self._btn_cls, button_name)
                self._ms.press(button)
            except Exception:
                pass
        elif self._backend == "ydotool":
            btn_map = {"left": "0x40001", "right": "0x40002", "middle": "0x40003"}
            subprocess.run(
                [
                    "ydotool",
                    "click",
                    "--clearmodifiers",
                    btn_map.get(button_name, "0x40001"),
                ],
                check=False,
            )

    def _mouse_release(self, button_name: str) -> None:
        if self._backend == "pynput":
            try:
                button = getattr(self._btn_cls, button_name)
                self._ms.release(button)
            except Exception:
                pass
        elif self._backend == "ydotool":
            btn_map = {"left": "0x40002", "right": "0x40008", "middle": "0x40020"}
            subprocess.run(
                [
                    "ydotool",
                    "click",
                    "--clearmodifiers",
                    btn_map.get(button_name, "0x40002"),
                ],
                check=False,
            )

    def _scroll(self, direction: str, amount: int) -> None:
        if self._backend == "pynput":
            try:
                dx, dy = (0, amount) if direction == "up" else (0, -amount)
                self._ms.scroll(dx, dy)
            except Exception:
                pass
        elif self._backend == "ydotool":
            axis = "1" if direction == "up" else "-1"
            for _ in range(amount):
                subprocess.run(["ydotool", "scroll", "--axis", axis], check=False)
