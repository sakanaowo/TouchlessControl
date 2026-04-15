# Gesture Vocabulary — Milestone 2 (Mouse Control)

Minimal 5-class vocabulary optimized for touchless mouse control.
Feature vector: 93-dim (63 3D-coords + 15 joint-angles + 5 tip-wrist + 5 tip-palm + 5 finger-state).

---

## Class Table

| Index | Name           | OS Action                | Trigger Policy                                |
| ----- | -------------- | ------------------------ | --------------------------------------------- |
| 0     | `null`         | None (Idle)              | No action ever emitted                        |
| 1     | `pointer_move` | Move cursor              | Bypasses StateMachine — continuous each frame |
| 2     | `left_click`   | Left mouse click         | `start` only via StateMachine — one-shot      |
| 3     | `drag_hold`    | Hold left button (drag)  | `start`/`end` via StateMachine — mouse_drag   |
| 4     | `scroll_mode`  | Scroll up/down (Y-delta) | Bypasses StateMachine — continuous each frame |

---

## Hand Shape Reference

| Name           | Extended fingers | Thumb     | Key features                                                  |
| -------------- | ---------------- | --------- | ------------------------------------------------------------- |
| `null`         | any casual rest  | any       | No deliberate gesture; hand relaxed, transitioning, or absent |
| `pointer_move` | index only       | in        | Only index finger straight; middle/ring/pinky curled          |
| `left_click`   | index + thumb    | tip-touch | Quick pinch: thumb-tip meets index-tip; others loosely curled |
| `drag_hold`    | none             | in        | Fist: all fingers curled tightly, thumb wrapped               |
| `scroll_mode`  | index + middle   | in        | V-sign: two fingers extended, palm facing camera              |

---

## Disambiguation — At-Risk Pairs

| Pair                            | Risk   | Discriminating features                    | Mitigation                                                      |
| ------------------------------- | ------ | ------------------------------------------ | --------------------------------------------------------------- |
| `pointer_move` vs `scroll_mode` | HIGH   | middle finger extension; z-depth of middle | Collect both at similar distances; z-depth is key discriminator |
| `left_click` vs `pointer_move`  | MEDIUM | thumb-index tip distance                   | left_click = tips touching; pointer_move = tips apart           |
| `drag_hold` vs `null` (rest)    | MEDIUM | intentionality; finger tightness           | Fist must be tight/deliberate; null = loose/relaxed             |

---

## Collection Guidelines

- **Samples per class**: ≥300 frames
- **Angles / rotations per class**:
  - Facing camera directly
  - 30° yaw left / right
  - 30° tilt up / down
  - Two hand-to-camera distances (near ~40cm, far ~70cm)
- **pointer_move**: Collect while moving hand in various directions
- **null**: Hand at rest, transitioning, partially formed gestures, entering/leaving frame
- **Data collection**: Use Tab → select class → Enter in guided collection UI

## T2.2 Blocker Note

`app.py` currently maps keyboard `0`–`9` → class index 0–9 only (`select_mode` function).
Classes 10 (`ok_sign`), 11 (`gun_sign`), 12 (`call_sign`) are unreachable via k-mode.
**Resolution** (part of T2.2): extend `select_mode()` to map keys `a` → 10, `b` → 11, `c` → 12.

---

## Status

- [ ] Pairs verified physically (walking-test: show each gesture, check confidence of current v1 model — only for orientation; new model will be trained after collection)
- [ ] Extended key mapping for classes 10–12 added (T2.2 pre-req)
- [ ] Data collection started (T2.2)
