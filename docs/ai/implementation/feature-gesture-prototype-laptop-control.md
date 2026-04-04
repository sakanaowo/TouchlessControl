---
phase: implementation
title: Implementation Guide
description: Prototype — Nâng cấp hand gesture recognition cho laptop control (Milestone 1)
feature: gesture-prototype-laptop-control
milestone: 1
---

# Implementation Guide

## Development Setup

**Prerequisites:**

```bash
# Activate existing virtualenv
source .venv/bin/activate  # or conda activate

# Install new dependencies
pip install pynput pyyaml

# Verify MediaPipe version
python -c "import mediapipe; print(mediapipe.__version__)"  # need >= 0.10
```

**New files to create:**

```
utils/
  feature_extractor.py        # T1.1
  gesture_state_machine.py    # T4.1
  action_mapper.py            # T5.2

model/keypoint_classifier/
  keypoint_classifier_v2.py   # T3.4
  keypoint_classifier_v2.tflite  # T3.3 (generated)
  keypoint_classifier_v2_label.csv  # 13 class names

config/
  gesture_actions.yaml        # T5.1
  gesture_vocabulary.md       # T2.1 notes
```

---

## Code Structure

### `utils/feature_extractor.py`

```python
import copy
import itertools
import math
import numpy as np

class FeatureExtractor:
    PALM_KP_INDICES = [0, 5, 9, 13, 17]
    # Each entry: [base, joint1, joint2, joint3, tip]
    # range(len-2) = range(3) → 3 angles per finger (at joint1, joint2, joint3) = 15 total
    FINGER_JOINTS = [
        [0, 5, 6, 7, 8],    # Index  (wrist→MCP→PIP→DIP→TIP)
        [0, 9, 10, 11, 12],  # Middle
        [0, 13, 14, 15, 16], # Ring
        [0, 17, 18, 19, 20], # Pinky
        [0, 1, 2, 3, 4],     # Thumb  (wrist→CMC→MCP→IP→TIP)
    ]

    def extract(self, landmark_list):
        """
        Args:
            landmark_list: list of 21 × [x, y, z] (raw pixel or 0-1 normalized from MediaPipe)
        Returns:
            list of 93 float32 values
        """
        kp = np.array(landmark_list, dtype=np.float32)  # (21, 3)

        # Block 1: relative + normalized coordinates (63 features)
        kp_rel = kp - kp[0]  # wrist as origin
        max_dist = np.max(np.abs(kp_rel)) + 1e-6
        kp_norm = (kp_rel / max_dist).flatten().tolist()  # 63 features

        # Block 2: joint angles (15 features)
        angles = self._compute_joint_angles(kp)

        # Block 3: fingertip to wrist distances (5 features)
        tips = [4, 8, 12, 16, 20]
        tip_wrist = [float(np.linalg.norm(kp[t] - kp[0])) / (max_dist + 1e-6) for t in tips]

        # Block 4: fingertip to palm center distances (5 features)
        palm_center = kp[self.PALM_KP_INDICES].mean(axis=0)
        tip_palm = [float(np.linalg.norm(kp[t] - palm_center)) / (max_dist + 1e-6) for t in tips]

        # Block 5: finger state booleans (5 features) — 1=extended, 0=bent
        finger_states = self._compute_finger_states(kp)

        return kp_norm + angles + tip_wrist + tip_palm + finger_states  # 63+15+5+5+5 = 93

    def _compute_joint_angles(self, kp):
        angles = []
        for joints in self.FINGER_JOINTS:
            for i in range(len(joints) - 2):
                a, b, c = kp[joints[i]], kp[joints[i+1]], kp[joints[i+2]]
                v1 = a - b
                v2 = c - b
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                angles.append(float(np.clip(cos_angle, -1.0, 1.0)))
        return angles  # 5 fingers × 3 angles = 15 (angle at joint[1], joint[2], joint[3] for each finger)

    def _compute_finger_states(self, kp):
        tips = [8, 12, 16, 20]  # Index through Pinky tips
        pips = [6, 10, 14, 18]  # Corresponding PIP joints
        states = []
        for tip, pip in zip(tips, pips):
            # Extended if tip is further from wrist than PIP
            states.append(1.0 if np.linalg.norm(kp[tip] - kp[0]) > np.linalg.norm(kp[pip] - kp[0]) else 0.0)
        # Thumb: compare tip to MCP
        states.append(1.0 if np.linalg.norm(kp[4] - kp[0]) > np.linalg.norm(kp[2] - kp[0]) else 0.0)
        return states  # 5 values
```

> **Note**: Verify output length = 93 in unit test (T1.1). Joint angle count may depend on final FINGER_JOINTS indexing — reconcile during implementation.

---

### `utils/gesture_state_machine.py`

```python
import time
from dataclasses import dataclass, field
from typing import Literal, Optional
import numpy as np

GESTURE_LABELS = [
    "null", "open_palm", "fist", "pointer", "thumbs_up", "thumbs_down",
    "v_sign", "three_fingers", "four_fingers", "pinch", "ok_sign",
    "gun_sign", "call_sign",
]

@dataclass
class GestureEvent:
    gesture: str
    event_type: Literal["start", "end", "hold"]

@dataclass
class GestureStateMachine:
    confidence_threshold: float = 0.82
    activation_frames: int = 5
    deactivation_frames: int = 10
    debounce_seconds: float = 0.5

    _state: str = field(default="idle", init=False)
    _candidate_class: int = field(default=0, init=False)
    _candidate_frames: int = field(default=0, init=False)
    _deactivation_frames_count: int = field(default=0, init=False)
    _active_class: int = field(default=-1, init=False)
    _last_emitted_time: float = field(default=0.0, init=False)
    _no_hand_since: float = field(default=0.0, init=False)

    def update(self, class_id: int, scores: np.ndarray) -> Optional[GestureEvent]:
        confidence = float(scores[class_id])
        is_null = (class_id == 0)
        is_confident = confidence >= self.confidence_threshold
        now = time.time()
        self._no_hand_since = 0.0  # hand is present this frame

        if self._state == "idle":
            if not is_null and is_confident:
                self._state = "tracking"
                self._candidate_class = class_id
                self._candidate_frames = 1
            return None

        elif self._state == "tracking":
            if class_id == self._candidate_class and is_confident:
                self._candidate_frames += 1
                if self._candidate_frames >= self.activation_frames:
                    if now - self._last_emitted_time > self.debounce_seconds:
                        self._state = "active"
                        self._active_class = class_id
                        self._deactivation_frames_count = 0
                        self._last_emitted_time = now
                        return GestureEvent(GESTURE_LABELS[class_id], "start")
            else:
                self._state = "idle"
                self._candidate_frames = 0
            return None

        elif self._state == "active":
            if class_id != self._active_class or not is_confident:
                self._deactivation_frames_count += 1
                if self._deactivation_frames_count >= self.deactivation_frames:
                    ended = self._active_class
                    self._state = "idle"
                    self._active_class = -1
                    self._deactivation_frames_count = 0
                    return GestureEvent(GESTURE_LABELS[ended], "end")
            else:
                self._deactivation_frames_count = 0
                # Emit hold event for repeat-capable gestures
                if now - self._last_emitted_time >= self.debounce_seconds:
                    self._last_emitted_time = now
                    return GestureEvent(GESTURE_LABELS[class_id], "hold")
            return None

        return None

    def update_no_hand(self) -> None:
        """Call each frame when no hand landmarks are detected."""
        if self._state != "active":
            return  # IDLE/TRACKING: no state to preserve
        now = time.time()
        if self._no_hand_since == 0.0:
            self._no_hand_since = now
        elif now - self._no_hand_since >= 1.0:
            # Grace period expired — silent reset
            self._state = "idle"
            self._active_class = -1
            self._deactivation_frames_count = 0
            self._no_hand_since = 0.0
```

---

### `config/gesture_actions.yaml`

```yaml
# Gesture → OS action mapping
# on_start: action triggered when gesture is ACTIVATED
# on_end: action triggered when gesture ENDS (optional)

open_palm:
  on_start: key_press
  key: "XF86AudioPlay"

fist:
  on_start: key_press
  key: "XF86AudioMute"

thumbs_up:
  on_start: key_press
  key: "XF86AudioRaiseVolume"

thumbs_down:
  on_start: key_press
  key: "XF86AudioLowerVolume"

v_sign:
  on_start: scroll
  direction: up
  amount: 5

three_fingers:
  on_start: scroll
  direction: down
  amount: 5

four_fingers:
  on_start: key_press
  key: "XF86MonBrightnessUp"

pinch:
  on_start: mouse_click
  button: left

ok_sign:
  on_start: mouse_click
  button: right

gun_sign:
  on_start: key_combo
  keys: ["alt", "tab"]

call_sign:
  on_start: key_press
  key: "XF86AudioPrev"

null:
  # No action
pointer:
  # Cursor movement handled separately in app.py
```

---

## Integration Points

### Changes to `app.py`

Minimal surgical changes — wrap existing logic:

```python
# ADD imports
from utils.feature_extractor import FeatureExtractor
from utils.gesture_state_machine import GestureStateMachine, GESTURE_LABELS
from utils.action_mapper import ActionMapper
from model.keypoint_classifier import KeyPointClassifierV2

# ADD instantiation after existing classifiers
feature_extractor = FeatureExtractor()
gesture_state_machine = GestureStateMachine()
action_mapper = ActionMapper("config/gesture_actions.yaml")
keypoint_classifier_v2 = KeyPointClassifierV2()

# REPLACE in main loop:
if results.multi_hand_landmarks:
    features = feature_extractor.extract(landmark_list_3d)  # pass 3D landmarks
    class_id, scores = keypoint_classifier_v2(features)

    # pointer gesture: separate continuous path (unchanged from v1)
    if class_id == POINTER_CLASS_INDEX:
        # existing index-finger tracking logic here
        pass
    else:
        event = gesture_state_machine.update(class_id, scores)
        if event:
            action_mapper.handle(event)
else:
    gesture_state_machine.update_no_hand()
```

**Important**: `landmark_list` in current code is 2D (x, y). Need to also extract z from `hand_landmarks.landmark[i].z`. Update `calc_landmark_list()` or pass raw landmark to `FeatureExtractor`.

---

## Error Handling

- If `pynput` action fails (e.g., Wayland permissions): log warning, continue — do NOT crash pipeline
- `ActionMapper.__init__` checks `os.environ.get('WAYLAND_DISPLAY')` on startup — uses ydotool if set, pynput otherwise. No runtime fallback needed.
- If MediaPipe returns no landmarks for a frame: `GestureStateMachine` receives no update (pass null frame)
- If `gesture_actions.yaml` missing or malformed: `ActionMapper` raises `FileNotFoundError` at startup (fast fail)
- TFLite model file not found: existing behavior (exception at startup)

---

## Performance Considerations

- `FeatureExtractor.extract()` uses NumPy — avoid Python loops where possible
- `GestureStateMachine` is pure Python, O(1) — negligible
- `ActionMapper.handle()` is I/O (pynput) — runs async if hold actions needed (not in M1)
- Keep `history_length = 16` for PointHistoryClassifier unchanged (backward compat)

---

## Security Notes

- `pynput` sends keyboard/mouse events to the OS — this is the intended use case
- `gesture_actions.yaml` should not be loaded from user-untrusted paths
- No network calls, no file writes during inference loop
