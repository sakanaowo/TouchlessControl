---
phase: testing
title: Testing Strategy
description: Prototype — Nâng cấp hand gesture recognition cho laptop control (Milestone 1)
feature: gesture-prototype-laptop-control
milestone: 1
---

# Testing Strategy

## Test Coverage Goals

- **Unit tests**: 100% of new modules (`feature_extractor`, `gesture_state_machine`, `action_mapper`, `keypoint_classifier_v2`)
- **Integration tests**: Pipeline flow from raw landmarks → gesture event
- **End-to-end (manual)**: Mỗi gesture trigger đúng OS action, Null class không trigger

---

## Unit Tests

### `utils/feature_extractor.py`

- [ ] **FE-1**: Output shape chính xác — `len(features) == 93`
- [ ] **FE-2**: Không có `NaN` hoặc `Inf` trong output
- [ ] **FE-3**: Normalized coordinates nằm trong `[-1, 1]`
- [ ] **FE-4**: Wrist (kp[0]) → relative coord = (0, 0, 0)
- [ ] **FE-5**: Open palm vs. fist có feature vector khác nhau rõ ràng (cosine sim < 0.8)
- [ ] **FE-6**: Xoay tay 90° → feature vector tương tự (scale-invariant test)
- [ ] **FE-7**: `finger_states` = [1,1,1,1,1] khi tất cả ngón duỗi thẳng
- [ ] **FE-8**: `finger_states` = [0,0,0,0,0] khi tất cả ngón gập lại (fist)

### `utils/gesture_state_machine.py`

- [ ] **SM-1**: Khởi tạo → state = `"idle"`
- [ ] **SM-2**: Null class input liên tục → vẫn giữ `"idle"`, không emit event
- [ ] **SM-3**: < `ACTIVATION_FRAMES` frames cùng class → không emit, state = `"tracking"`
- [ ] **SM-4**: `ACTIVATION_FRAMES` frames cùng class, confidence > threshold → emit `GestureEvent(gesture, "start")`, state → `"active"`
- [ ] **SM-5**: Trong `"active"`, đổi class → `deactivation_frames_count` tăng dần
- [ ] **SM-6**: `DEACTIVATION_FRAMES` frames khác class → emit `GestureEvent(gesture, "end")`, state → `"idle"`
- [ ] **SM-7**: Debounce — cùng gesture emit lại trong < 0.5s → không emit lần 2
- [ ] **SM-8**: Tracking bị interrupt (class thay đổi trước khi đủ frames) → reset về `"idle"`
- [ ] **SM-9**: Confidence dưới threshold ngay cả khi đúng class → không activate
- [ ] **SM-10**: `update_no_hand()` trong ACTIVE state, elapsed < 1s → giữ nguyên ACTIVE, không emit event
- [ ] **SM-11**: `update_no_hand()` liên tục trong ACTIVE state, elapsed ≥ 1s → reset về IDLE, không emit end event (silent)
- [ ] **SM-12**: Tay quay lại trong 1s grace period → `update()` reset `_no_hand_since`, ACTIVE tiếp tục bình thường
- [ ] **SM-13**: Trong ACTIVE, debounce elapsed + cùng class + conf > θ → emit `GestureEvent(gesture, "hold")`
- [ ] **SM-14**: `update_no_hand()` trong IDLE → không thay đổi gì

### `model/keypoint_classifier/keypoint_classifier_v2.py`

- [ ] **KC-1**: Load model không raise exception
- [ ] **KC-2**: Output là `(int, np.ndarray)` — class index và scores array shape `(13,)`
- [ ] **KC-3**: `sum(scores) ≈ 1.0` (softmax output)
- [ ] **KC-4**: `scores[class_index] == max(scores)`
- [ ] **KC-5**: Input shape mismatch → raise `ValueError` rõ ràng

### `utils/action_mapper.py`

- [ ] **AM-1**: Load valid YAML → không raise exception
- [ ] **AM-2**: `handle(GestureEvent("null", "start"))` → không gọi pynput
- [ ] **AM-3**: `handle(GestureEvent("fist", "start"))` → gọi đúng pynput action (mock)
- [ ] **AM-4**: Unknown gesture name → log warning, không crash
- [ ] **AM-5**: pynput failure → log error, không crash pipeline
- [ ] **AM-6**: `repeat: true` gesture nhận `GestureEvent("thumbs_up", "hold")` → gọi đúng pynput action (mock)
- [ ] **AM-7**: `repeat: false` gesture nhận `GestureEvent("fist", "hold")` → không gọi pynput (bỏ qua hold)
- [ ] **AM-8**: `WAYLAND_DISPLAY` set → ActionMapper dùng ydotool (mock subprocess), không dùng pynput

---

## Integration Tests

- [ ] **INT-1**: `FeatureExtractor` → `KeyPointClassifierV2` → output hợp lệ cho synthetic landmark
- [ ] **INT-2**: 5 frames synthetic "thumbs_up" → `GestureStateMachine` emits `GestureEvent("thumbs_up", "start")`
- [ ] **INT-3**: Null class frames sau "thumbs_up" → emits `GestureEvent("thumbs_up", "end")`
- [ ] **INT-4**: Rapid class switching (< ACTIVATION_FRAMES each) → no event emitted
- [ ] **INT-5**: `app.py --no-actions` chạy 100 frames synthetic → không crash, FPS > 25

---

## End-to-End Tests (Manual)

Chạy `python app.py --no-actions` trước để validate recognition mà không có OS side effects.

### Checklist per gesture

| Gesture       | Nhận dạng đúng | State = ACTIVE khi cần | State = IDLE khi nghỉ | Action map đúng |
| ------------- | -------------- | ---------------------- | --------------------- | --------------- |
| null          | ✓ / ✗          | N/A                    | ✓ / ✗                 | N/A             |
| open_palm     | ✓ / ✗          | ✓ / ✗                  | ✓ / ✗                 | ✓ / ✗           |
| fist          | ✓ / ✗          | ✓ / ✗                  | ✓ / ✗                 | ✓ / ✗           |
| pointer       | ✓ / ✗          | ✓ / ✗                  | ✓ / ✗                 | ✓ / ✗           |
| thumbs_up     | ✓ / ✗          | ✓ / ✗                  | ✓ / ✗                 | ✓ / ✗           |
| thumbs_down   | ✓ / ✗          | ✓ / ✗                  | ✓ / ✗                 | ✓ / ✗           |
| v_sign        | ✓ / ✗          | ✓ / ✗                  | ✓ / ✗                 | ✓ / ✗           |
| three_fingers | ✓ / ✗          | ✓ / ✗                  | ✓ / ✗                 | ✓ / ✗           |
| four_fingers  | ✓ / ✗          | ✓ / ✗                  | ✓ / ✗                 | ✓ / ✗           |
| pinch         | ✓ / ✗          | ✓ / ✗                  | ✓ / ✗                 | ✓ / ✗           |
| ok_sign       | ✓ / ✗          | ✓ / ✗                  | ✓ / ✗                 | ✓ / ✗           |
| gun_sign      | ✓ / ✗          | ✓ / ✗                  | ✓ / ✗                 | ✓ / ✗           |
| call_sign     | ✓ / ✗          | ✓ / ✗                  | ✓ / ✗                 | ✓ / ✗           |

**Pass criteria**: tất cả dòng ✓ trước khi đánh dấu Milestone 1 Done.

---

## Test Data

- **Unit test fixtures**: synthetic landmark arrays (21 × [x,y,z]) được hard-code trong `tests/fixtures/landmarks.py`
  - `OPEN_PALM_LANDMARKS`: ngón duỗi thẳng
  - `FIST_LANDMARKS`: ngón gập
  - `THUMBS_UP_LANDMARKS`: chỉ ngón cái duỗi
- **Integration mocks**: mock `pynput.keyboard.Controller` và `pynput.mouse.Controller`

**Run tests:**

```bash
pytest tests/ -v --tb=short
pytest tests/ --cov=utils --cov=model --cov-report=term-missing
```

---

## Manual Testing — Acceptance Criteria

Before marking Milestone 1 complete:

- [ ] FPS ≥ 25 khi chạy full pipeline (`--no-actions`)
- [ ] Null class: tay nghỉ tự nhiên trong 30 giây → 0 OS actions triggered
- [ ] Mỗi gesture: 10 lần thử, ≥ 9/10 nhận dạng đúng và trigger action đúng
- [ ] Chuyển nhanh giữa 2 gesture (< 0.5s) → không trigger nhầm
- [ ] `--no-actions` flag: pipeline chạy bình thường, không có OS side effects

---

## Performance Testing

```bash
# Measure pipeline FPS
python app.py --no-actions 2>&1 | grep FPS | tail -20
```

Target: ≥ 25 FPS (average over 60 frames)

---

## Bug Tracking

- Severity 1 (blocker): false positive OS action (system misbehaves)
- Severity 2 (major): gesture not recognized > 20% of tries
- Severity 3 (minor): overlay display glitch, FPS drops occasionally
- Regression: sau mỗi thay đổi code, re-run `pytest tests/` trước khi test manual
