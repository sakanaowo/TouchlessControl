# Gesture Vocabulary — Milestone 1

Locked vocabulary for the 13-class classifier (12 gesture classes + 1 Null class).
Feature vector: 93-dim (63 3D-coords + 15 joint-angles + 5 tip-wrist + 5 tip-palm + 5 finger-state).

---

## Class Table

| Index | Name            | OS Action             | Trigger Policy                          |
| ----- | --------------- | --------------------- | --------------------------------------- |
| 0     | `null`          | None (Idle)           | No action ever emitted                  |
| 1     | `open_palm`     | Pause / Play media    | `start` only — one-shot                 |
| 2     | `fist`          | Mute toggle           | `start` only — one-shot                 |
| 3     | `pointer`       | Move cursor (legacy)  | Bypasses StateMachine — continuous path |
| 4     | `thumbs_up`     | Volume up             | `start` + `hold` — repeat               |
| 5     | `thumbs_down`   | Volume down           | `start` + `hold` — repeat               |
| 6     | `v_sign`        | Scroll up             | `start` + `hold` — repeat               |
| 7     | `three_fingers` | Scroll down           | `start` + `hold` — repeat               |
| 8     | `four_fingers`  | Brightness up         | `start` + `hold` — repeat               |
| 9     | `pinch`         | Left click            | `start` only — one-shot                 |
| 10    | `ok_sign`       | Right click           | `start` only — one-shot                 |
| 11    | `gun_sign`      | Next window / Alt+Tab | `start` only — one-shot                 |
| 12    | `call_sign`     | Previous track        | `start` only — one-shot                 |

---

## Hand Shape Reference

| Name            | Extended fingers  | Thumb     | Key features                                                                                                     |
| --------------- | ----------------- | --------- | ---------------------------------------------------------------------------------------------------------------- |
| `null`          | any casual rest   | any       | Low confidence output from model; tay buông tự nhiên, di chuyển giữa gesture                                     |
| `open_palm`     | all 5             | yes       | All fingers straight, spread open, facing camera                                                                 |
| `fist`          | none              | in        | All fingers curled tightly, thumb wrapped                                                                        |
| `pointer`       | index             | in        | Only index straight; middle/ring/pinky curled                                                                    |
| `thumbs_up`     | thumb             | up        | Thumb up, all other fingers curled; knuckles toward camera                                                       |
| `thumbs_down`   | thumb             | down      | Thumb down; hand rotated 180° vs thumbs_up                                                                       |
| `v_sign`        | index + middle    | in        | V shape, palm facing camera; middle finger slightly bent at DIP (key discriminator vs pointer — z-depth differs) |
| `three_fingers` | index+middle+ring | in        | Three fingers extended, pinky + thumb curled                                                                     |
| `four_fingers`  | index–pinky       | in        | Four fingers extended, thumb tucked in; vs open_palm: thumb clearly absent                                       |
| `pinch`         | index + thumb     | tip-touch | Index-tip meets thumb-tip forming circle; others loosely curled                                                  |
| `ok_sign`       | middle–pinky      | tip-touch | Index-tip meets thumb-tip like pinch, BUT middle/ring/pinky extended straight (discriminator vs pinch)           |
| `gun_sign`      | index + thumb     | up        | Index points forward, thumb up (L shape); vs thumbs_up: index also extended                                      |
| `call_sign`     | thumb + pinky     | out       | Thumb and pinky extended, middle three curled ("phone" shape); vs gun_sign: pinky replaces index                 |

---

## Disambiguation — At-Risk Pairs

These pairs must be physically tested before full data collection (T2.1 requirement).
If confusion persists during test collection, apply the listed mitigation.

| Pair                          | Risk   | Discriminating features                    | Decision / Mitigation                                                                                        |
| ----------------------------- | ------ | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| `pointer` vs `v_sign`         | HIGH   | z-depth of middle finger; middle DIP angle | Primary discriminator: middle-finger z-depth (design decision). Accepted. Collect both at similar distances. |
| `thumbs_up` vs `gun_sign`     | HIGH   | index finger extension                     | Must clearly extend index for gun_sign. Collect gun_sign from side angle too.                                |
| `pinch` vs `ok_sign`          | MEDIUM | middle/ring/pinky state                    | ok_sign requires 3 fingers clearly extended. Show from slight angle for better PIP visibility.               |
| `open_palm` vs `four_fingers` | MEDIUM | thumb state                                | Emphasize tucked thumb for four_fingers. Use multiple hand-tilt angles.                                      |
| `fist` vs `null` (rest)       | MEDIUM | intentionality                             | Collect null class with partially curled hand — fist must be tight and deliberate.                           |
| `thumbs_down` vs `thumbs_up`  | LOW    | orientation                                | Wrist rotation axis will differ in normalized coords; typically stable.                                      |
| `call_sign` vs `gun_sign`     | LOW    | pinky vs index                             | Both have thumb up; extended finger identity (index vs pinky) differs clearly.                               |
| `three_fingers` vs `v_sign`   | LOW    | ring finger                                | Ring finger extension; collect both at close-to-camera angles.                                               |

---

## Collection Guidelines for T2.2

- **Samples per class**: ~150 (non-null), ~200 (null)
- **Angles / rotations to cover per class** (per session in app mode `k`):
  - Facing camera directly
  - 30° yaw left / right
  - 30° tilt up / down
  - Two hand-to-camera distances (near ~40cm, far ~70cm)
- **Null samples (class 0)**:
  - Hand at rest, fingers loose
  - Hand mid-transition between gestures (motion blur frame)
  - Partially formed gestures (incomplete attempt)
  - Hand entering / leaving frame edge
- **Data entry in app**: press key `k` → number key `0`–`9` selects class (app currently only supports 0–9; class 10–12 need `logging_csv` to support number keys `a`/`b`/`c` or multi-digit input — see T2.2 blocker note below)

## T2.2 Blocker Note

`app.py` currently maps keyboard `0`–`9` → class index 0–9 only (`select_mode` function).
Classes 10 (`ok_sign`), 11 (`gun_sign`), 12 (`call_sign`) are unreachable via k-mode.
**Resolution** (part of T2.2): extend `select_mode()` to map keys `a` → 10, `b` → 11, `c` → 12.

---

## Status

- [ ] Pairs verified physically (walking-test: show each gesture, check confidence of current v1 model — only for orientation; new model will be trained after collection)
- [ ] Extended key mapping for classes 10–12 added (T2.2 pre-req)
- [ ] Data collection started (T2.2)
