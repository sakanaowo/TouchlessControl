---
phase: requirements
title: Requirements & Problem Understanding
description: Prototype — Nâng cấp hand gesture recognition cho laptop control (Milestone 1)
feature: gesture-prototype-laptop-control
milestone: 1
---

# Requirements & Problem Understanding

## Problem Statement

**What problem are we solving?**

Hệ thống nhận dạng cử chỉ tay hiện tại (`app.py`) có giới hạn nghiêm trọng khi muốn dùng để điều khiển laptop:

- Chỉ nhận dạng được **3 static gesture** (Open / Close / Pointer) — không đủ để map thành các thao tác laptop đa dạng
- Dynamic classifier chỉ track **1 điểm duy nhất** (đầu ngón trỏ) — không capture được multi-finger pattern
- **Không có Null/Idle class** — model luôn ra output ngay cả khi tay đứng yên hoặc không có gesture
- **Không có temporal segmentation** — không biết gesture bắt đầu/kết thúc lúc nào, dẫn đến trigger lặp hoặc bỏ sót
- Bỏ qua **z-depth** do MediaPipe cung cấp, giảm discriminative power

**Who is affected?**

Developer (solo, thử nghiệm) muốn điều khiển laptop bằng tay qua webcam thông thường.

**Current workaround?**

Không có — hệ thống chưa đủ khả năng để dùng thực tế cho laptop control.

---

## Goals & Objectives

**Primary goals (Milestone 1 — Prototype):**

- Mở rộng gesture vocabulary từ 3 lên **≥ 12 static gesture** đa ngón
- Thêm **Null/Idle class** để ngăn false positive khi tay đứng yên
- Implement **Gesture State Machine** để phát hiện start/stop của gesture
- Map gesture → **pynput actions** (keyboard/mouse) để điều khiển laptop thực sự
- Giữ nguyên latency tổng **< 50ms** (30fps)

**Secondary goals:**

- Thêm feedback trực quan trên overlay: tên gesture + confidence + state
- Thiết kế gesture vocabulary tránh xung đột (disambiguation)

**Non-goals (explicitly out of scope cho Milestone 1):**

- Temporal/dynamic gesture recognition (dùng Approach B — để Milestone 2)
- Multi-hand support
- Custom training pipeline UI
- Deployment / packaging
- bất kỳ thay đổi nào với MediaPipe detector

---

## User Stories & Use Cases

**Core stories:**

- As a user, I want to raise my thumb to increase laptop volume, so I don't need to touch the keyboard
- As a user, I want to make a fist to mute audio, so I can quickly silence sound
- As a user, I want to hold up an open palm to pause media playback
- As a user, I want to point with only the index finger and move it to control the cursor
- As a user, I want the system to do nothing when my hand is resting or moving between gestures (Null class)
- As a user, I want the system to recognize a gesture only after I hold it clearly for ~150ms (no accidental triggers)
- As a developer, I want to easily add a new gesture by collecting ~100 samples and retraining, without changing architecture

**Key workflows:**

1. User shows gesture → system detects after 5-frame hold → triggers OS action → resets state
2. User moves hand naturally into frame → system stays in Idle state (Null class fires) → no spurious action
3. User holds gesture ambiguously → confidence < threshold → system waits, no action

**Edge cases:**

- Partial occlusion (finger bent, not fully visible)
- Fast hand movement between gestures (Null transition)
- Lighting change mid-session
- V-sign vs Pointer confusion — distinguished via z-depth + middle-finger joint angle (DIP); accepted noise from RGB pseudo-depth
- **Hand leaves frame mid-gesture** — system preserves ACTIVE state for 1s; if hand returns within 1s, resumes; otherwise resets to IDLE. Actions are NOT dispatched while hand is absent.
- `pointer` gesture is a **special continuous cursor-control mode** — it bypasses GestureStateMachine and ActionMapper entirely, handled via existing landmark tracking path

---

## Success Criteria

| Criterion                                            | Target                                       |
| ---------------------------------------------------- | -------------------------------------------- |
| Number of static gesture classes                     | ≥ 12 (+ 1 Null)                              |
| Per-class accuracy on test set                       | ≥ 88%                                        |
| False positive rate (Null class acting)              | < 5%                                         |
| Gesture activation latency (from hold to action)     | < 200ms (5 frames @ 30fps ≈ 167ms)           |
| End-to-end pipeline latency                          | < 50ms/frame                                 |
| Correct OS action triggered per gesture              | ≥ 9/10 tries per gesture (manual validation) |
| Debounce — same gesture not re-triggered within 0.5s | Verified                                     |

---

## Constraints & Assumptions

**Technical constraints:**

- Hardware: CPU-only laptop, no GPU inference
- MediaPipe Hands: fixed at 21 landmarks × 3D — cannot change upstream model
- Inference runtime: TFLite (can migrate to ONNX if faster)
- OS: Linux — **Wayland** session (pynput may not work; fallback: `ydotool` or `python-evdev`)
- Camera: RGB webcam, 30fps, 960×540
- z-depth from MediaPipe is pseudo-depth (relative, not metric) — used as feature but treated as noisy signal; feature vector = **93-dim**

**Implementation constraints:**

- Milestone 1 must not require a large dataset — max ~100–200 samples/class (collectable in 1 day)
- Changes to `app.py` must be backward-compatible (no pipeline rearchitecture in M1)
- New classifier must be exportable to `.tflite`

**Assumptions:**

- User performs gestures clearly and deliberately (not sign language fluency required)
- OS input control will use `pynput` first; if blocked by Wayland, fallback to `ydotool` (must be installed)
- Lighting conditions are reasonable (indoor, visible hand)
- **Activating/pausing gesture dispatch**: đưa tay ra ngoài frame = dispatch tạm dừng tự động (không cần hotkey trong M1). ESC = tắt app.
- `pointer` (cursor control) is a legacy continuous mode separate from the new gesture pipeline — no changes to its internal behavior in M1

---

## Questions & Open Items

> All M1 questions resolved. Decisions recorded below.

- [x] **Gesture vocabulary**: 12 static classes confirmed (see Design doc). v_sign + pointer coexist — distinguished by z-depth + middle-finger DIP joint angle. Final vocabulary list locked in `config/gesture_vocabulary.md` after T2.1 physical test.
- [x] **Cursor control (pointer gesture)**: STAYS in existing landmark tracking path. Does NOT go through GestureStateMachine or ActionMapper. No changes to pointer behavior in M1.
- [x] **pynput vs xdotool on Linux**: OS = Wayland. Primary: `pynput`. If blocked by compositor permissions, fallback: `ydotool` (Wayland-native). Test in T5.3 before integration.
- [x] **z-depth as feature**: YES — use all 3 components (x, y, z) from MediaPipe. Accept pseudo-depth noise. Feature vector = 93-dim. This is the key discriminator for v_sign vs pointer.
- [x] **Hand leaves frame**: State preserved for 1s (no dispatch while absent). If hand returns within 1s → resume ACTIVE state. After 1s → reset to IDLE.
- [x] **Activating/pausing dispatch**: Tay ra ngoài frame = dispatch tự tắt. No explicit toggle hotkey in M1.

### Open for M2+

- [ ] Dynamic gesture vocabulary for Transformer-based classifier (Milestone 2)
- [ ] Multi-user calibration (gesture shapes vary by hand size)
- [ ] Packaging and auto-start on login
