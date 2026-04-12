---
phase: planning
title: Guided Gesture Collection UI — Planning
description: Task breakdown for implementing session-based collection with visual feedback
---

# Planning — Guided Gesture Collection UI

## Milestones

- [x] **M1**: Camera MJPG 720p + CollectionManager FSM (core logic) ✅ 2026-04-10
- [x] **M2**: Collection overlay & progress UI ✅ 2026-04-10
- [x] **M3**: Balance indicator + polish ✅ 2026-04-10

## Task Breakdown

### Phase 1: Camera & Core Logic

- [x] **T1.1** — MJPG 720p camera setup ✅
  - Set `CAP_PROP_FOURCC` to MJPG before setting resolution
  - Default `--width 1280 --height 720`
  - Fallback: if actual resolution ≠ requested, log warning and continue
  - **Files**: `app.py` (get_args + cap setup)

- [x] **T1.2** — Create `utils/class_menu.py` ✅
  - `ClassMenu` class: load labels from `keypoint_classifier_label.csv`
  - ↑/↓ navigation, Enter confirm → returns class_id
  - `toggle()` via Tab key, `draw(image)` renders menu overlay
  - Hot-reload: re-read CSV mỗi lần toggle open
  - **Files**: new `utils/class_menu.py`

- [x] **T1.3** — Create `utils/collection_manager.py` ✅
  - `CollectionSession` dataclass (class_id, target_count, collected, timeout)
  - `CollectionManager` FSM: idle → countdown → recording → done → idle
  - RAM buffer: `_buffer` list, flush on success, discard on cancel
  - Quality gate: accept frame only if hand confidence ≥ 0.7
  - Frame skip: `frame_skip=2`, ghi 1 frame mỗi N frame
  - Multi-hand: `on_frame(hands: list[HandData]) → int`, ghi cả 2 tay
  - Timeout: 10s → flush partial + thông báo
  - `collected` đếm frames (không phải rows)
  - Auto-count existing samples from CSV on init
  - **Files**: new `utils/collection_manager.py`

- [x] **T1.4** — Unit tests for ClassMenu + CollectionManager ✅ (62 pass)
  - ClassMenu: load labels, navigation, confirm, toggle
  - CollectionManager: state transitions, quality gate, cancel → discard, auto-stop, timeout → flush partial, frame skip, multi-hand counting
  - **Files**: new `tests/unit/test_class_menu.py`, `tests/unit/test_collection_manager.py`

### Phase 2: Integration & Overlay

- [x] **T2.1** — Wire ClassMenu + CollectionManager into main loop ✅
  - Replace `current_class` / `logging_csv` latch with ClassMenu + CollectionManager
  - Key handling: Tab → menu toggle, ↑/↓ → navigate, Enter → start_session, Esc/Space → cancel
  - Bỏ phím `n` toggle mode
  - Frame handling: pass hands list to `on_frame()`, log if accepted
  - **Files**: `app.py` (main loop)

- [x] **T2.2** — Countdown overlay ✅
  - Large centered text: `3... 2... 1...` with class name below
  - Semi-transparent background rectangle for readability
  - **Files**: `app.py` (new `draw_countdown()` function)

- [x] **T2.3** — Recording overlay + progress bar ✅
  - Top banner: `[REC ●] CLASS:4 thumbs_up  12/30`
  - Progress bar under banner (green fill)
  - Green border around frame while recording
  - **Files**: `app.py` (update `draw_info()` or new `draw_recording_overlay()`)

- [x] **T2.4** — Done notification ✅
  - `✓ 30 samples saved for class 4 (thumbs_up)` — hiện 1.5s rồi fade
  - **Files**: `app.py`

### Phase 3: Balance & Polish

- [x] **T3.1** — Class balance indicator ✅
  - Read class counts from CollectionManager
  - Draw mini bar chart ở góc phải frame
  - Highlight class dưới target (< 100 samples)
  - **Files**: `app.py` (new `draw_balance_chart()`)

- [x] **T3.2** — Configurable batch size ✅
  - `+`/`-` keys hoặc Shift+number để đổi batch size (10/30/50/100)
  - Show current batch size on idle overlay
  - **Files**: `app.py`, `utils/collection_manager.py`

- [ ] **T3.3** — Diversity check (deferred — optional)
  - Skip frame if landmark delta < ε vs previous accepted frame
  - Prompt "Di chuyển tay để tăng diversity" nếu bị skip nhiều
  - **Files**: `utils/collection_manager.py`

## Dependencies

```
T1.1 (camera) ─────┐
                    ├──→ T2.1 (integration) ──→ T2.2, T2.3, T2.4 ──→ T3.1, T3.2
T1.2 (class menu) ──┤
T1.3 (manager) ─────┘
T1.4 (tests) ← T1.2 + T1.3
T3.3 (diversity) ← T2.1
```

- T1.1, T1.2, T1.3 song song
- T1.4 phụ thuộc T1.2 + T1.3
- T2.x phụ thuộc T1.1 + T1.2 + T1.3

## Risks & Mitigation

| Risk                                  | Impact       | Mitigation                                |
| ------------------------------------- | ------------ | ----------------------------------------- |
| MJPG 720p giảm FPS do decode overhead | UI chậm      | Benchmark trước; fallback về 640×480      |
| MediaPipe chậm hơn ở 720p             | FPS < 20     | Dùng `model_complexity=0` nếu cần         |
| Overlay phức tạp che tay              | Khó quan sát | Đặt overlay ở rìa frame, semi-transparent |
| Người dùng không đọc hướng dẫn        | Dùng sai     | UI self-explanatory, minimal text         |

## Resources Needed

- Existing: OpenCV, MediaPipe, conda env `sign`
- No new dependencies required
