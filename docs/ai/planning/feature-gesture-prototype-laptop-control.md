---
phase: planning
title: Project Planning & Task Breakdown
description: Prototype — Nâng cấp hand gesture recognition cho laptop control (Milestone 1)
feature: gesture-prototype-laptop-control
milestone: 1
---

# Project Planning & Task Breakdown

## Milestones

- [x] **M0**: Research & architecture decision (hoàn thành — xem `docs/research/gesture-upgrade-laptop-control.md`)
- [ ] **M1 — Prototype**: MLP enhanced + Null class + GestureStateMachine + OS action mapping
- [ ] **M2**: Transformer-based temporal classifier (Milestone 2, scope riêng)
- [ ] **M3**: Scaling với TD-GCN (Milestone 3, sau khi có đủ data)

---

## Task Breakdown — Milestone 1

### Phase 1: Feature Engineering (Ngày 1 sáng)

- [ ] **T1.1** — Viết `utils/feature_extractor.py`
  - Function: normalize 21 KP relative to wrist, scale to [-1,1] (x, y, z included)
  - Function: compute 15 joint angles (5 fingers × 3 joints)
  - Function: compute 10 distances (5 tip→wrist, 5 tip→palm_center) in 3D
  - Function: compute 5 finger state booleans (bent/straight)
  - Unit test: verify output shape = (93,), no NaN
  - _Effort: 2h_

- [ ] **T1.2** — Cập nhật `app.py` để dùng `FeatureExtractor` thay `pre_process_landmark()`
  - Giữ backward compat: `pre_process_landmark()` vẫn còn nhưng không dùng trong main path
  - _Effort: 30min_

- [ ] **T1.3** — Verify pipeline vẫn chạy với feature mới (dry-run, chưa cần classifier mới)
  - _Effort: 30min_

### Phase 2: Data Collection (Ngày 1 chiều)

- [ ] **T2.1** — Thiết kế và confirm gesture vocabulary (13 classes + Null)
  - Kiểm tra từng gesture: có phân biệt đủ rõ ràng không? (không bị confuse với nhau)
  - Ghi chú vào `config/gesture_vocabulary.md`
  - _Effort: 1h_

- [ ] **T2.2** — Thu thập training data qua công cụ ghi sẵn trong `app.py` (mode `k`)
  - Mỗi class: **~150 samples** (giữ tay ở nhiều góc, khoảng cách, ánh sáng)
  - Null class: ~200 samples (tay nghỉ, giữa các gesture, chuyển động nhanh)
  - Tổng: ~2000–2200 samples
  - _Effort: 2–3h_

- [ ] **T2.3** — Split train/val (80/20), kiểm tra class balance
  - _Effort: 30min_

### Phase 3: Model Training (Ngày 2 sáng)

- [ ] **T3.1** — Cập nhật `keypoint_classification.ipynb`
  - Input shape: (93,) thay vì (42,)
  - Architecture: `Dense(256) → Dropout(0.3) → Dense(128) → Dropout(0.2) → Dense(13, softmax)`
  - Loss: categorical crossentropy + label smoothing 0.1
  - Callback: EarlyStopping patience=15, ReduceLROnPlateau
  - _Effort: 1.5h_

- [ ] **T3.2** — Train và evaluate
  - Target: val accuracy ≥ 88%, Null class precision ≥ 95%
  - Confusion matrix analysis — identify và fix confusing pairs
  - _Effort: 1h (+ 30min nếu cần collect thêm data)_

- [ ] **T3.3** — Export sang `keypoint_classifier_v2.tflite`
  - Verify output shape trả về confidence scores (không chỉ argmax)
  - _Effort: 30min_

- [ ] **T3.4** — Viết `model/keypoint_classifier/keypoint_classifier_v2.py`
  - Return `(class_index, softmax_scores)` thay vì chỉ `class_index`
  - _Effort: 30min_

### Phase 4: GestureStateMachine (Ngày 2 chiều)

- [ ] **T4.1** — Viết `utils/gesture_state_machine.py`
  - States: `idle`, `tracking`, `active`
  - Params: `CONFIDENCE_THRESHOLD=0.82`, `ACTIVATION_FRAMES=5`, `DEACTIVATION_FRAMES=10`, `DEBOUNCE_SECONDS=0.5`
  - Returns `GestureEvent(name, event_type)` hoặc `None`
  - _Effort: 2h_

- [ ] **T4.2** — Unit test state machine
  - Test: idle → null input → stays idle
  - Test: 5 frames same class > threshold → emits start event
  - Test: debounce — no re-emit within 0.5s
  - Test: transition idle → tracking → idle (confidence drops mid-sequence)
  - _Effort: 1h_

### Phase 5: Action Mapping (Ngày 3 sáng)

- [ ] **T5.1** — Tạo `config/gesture_actions.yaml`
  - Map 12 gesture class names → pynput actions
  - Schema: `gesture_name: {on_start: key_press|mouse_click, key|button: value}`
  - _Effort: 30min_

- [ ] **T5.2** — Viết `utils/action_mapper.py`
  - Đọc config YAML
  - `handle(event)` → gọi pynput
  - _Effort: 1h_

- [ ] **T5.3** — Thêm dependencies vào `requirements.txt`
  - `pynput` — primary OS input
  - `ydotool` — Wayland fallback (system package: `sudo apt install ydotool`)
  - Kiểm tra pynput: `python -c "from pynput.keyboard import Controller; Controller().press('a')"` trên session hiện tại
  - Nếu fail → switch ActionMapper sang ydotool subprocess mode
  - _Effort: 45min_

### Phase 6: Integration & UI (Ngày 3 chiều)

- [ ] **T6.1** — Wire tất cả components trong `app.py`
  - `FeatureExtractor` → `KeyPointClassifierV2` → `GestureStateMachine` → `ActionMapper`
  - Thêm `--no-actions` flag để chạy xem gesture mà không trigger OS
  - _Effort: 1.5h_

- [ ] **T6.2** — Cập nhật overlay
  - Hiển thị: gesture name + confidence (%) + state badge (IDLE/TRACKING/ACTIVE)
  - Color code: xanh = ACTIVE, vàng = TRACKING, xám = IDLE
  - _Effort: 1h_

- [ ] **T6.3** — Smoke test toàn pipeline
  - Chạy với `--no-actions`, verify mọi gesture được nhận dạng đúng
  - Enable actions, test từng gesture thực tế trên OS
  - _Effort: 1h_

### Phase 7: Polish & Docs (Ngày 4)

- [ ] **T7.1** — Fine-tune thresholds nếu cần (sau smoke test)
  - `CONFIDENCE_THRESHOLD`, `ACTIVATION_FRAMES`, `DEBOUNCE_SECONDS`
  - _Effort: 1–2h_

- [ ] **T7.2** — Điền `docs/ai/implementation/feature-gesture-prototype-laptop-control.md`
  - _Effort: 30min_

- [ ] **T7.3** — Điền `docs/ai/testing/feature-gesture-prototype-laptop-control.md` và chạy test suite
  - _Effort: 1h_

- [ ] **T7.4** — Update `README_EN.md` với danh sách gesture mới và hướng dẫn sử dụng
  - _Effort: 30min_

---

## Dependencies

```
T1.1 (feature extractor)
  └── T1.2 (integrate to app.py)
        └── T1.3 (dry-run)
              └── T2.2 (collect data with new pipeline)
                    └── T2.3 (split)
                          └── T3.1–T3.4 (train + export)
                                └── T4.1–T4.2 (state machine, parallel with T3)
                                      └── T5.1–T5.3 (action mapping)
                                            └── T6.1–T6.3 (integration)
                                                  └── T7.1–T7.4 (polish)

T2.1 (vocabulary) ──► must be done before T2.2
T4.1 ──► can start in parallel with T3.1 (no model dependency)
T5.3 (pynput install) ──► before T5.2
```

**External dependencies:**

- `pynput` — Linux, requires X11 or Wayland with input permission
- `PyYAML` — for gesture_actions.yaml (likely already installed)
- MediaPipe >= 0.10 (existing)

---

## Timeline & Estimates

| Day                | Tasks     | Deliverable                                  |
| ------------------ | --------- | -------------------------------------------- |
| **Ngày 1** (sáng)  | T1.1–T1.3 | `FeatureExtractor` ready, pipeline dry-run ✓ |
| **Ngày 1** (chiều) | T2.1–T2.3 | 2000+ labeled samples collected              |
| **Ngày 2** (sáng)  | T3.1–T3.4 | Trained model v2, val acc ≥ 88%              |
| **Ngày 2** (chiều) | T4.1–T4.2 | `GestureStateMachine` + tests pass           |
| **Ngày 3** (sáng)  | T5.1–T5.3 | `ActionMapper` + OS actions work             |
| **Ngày 3** (chiều) | T6.1–T6.3 | Full pipeline integrated, smoke test         |
| **Ngày 4**         | T7.1–T7.4 | Fine-tuned, documented, tested               |

**Total estimate: 4 ngày** (có thể 3 ngày nếu data collection nhanh)

---

## Risks & Mitigation

| Risk                                        | Likelihood | Impact     | Mitigation                                                                            |
| ------------------------------------------- | ---------- | ---------- | ------------------------------------------------------------------------------------- |
| Gesture vocabulary có 2 class bị confuse    | Cao        | Trung bình | Test từng cặp trước khi collect full data (T2.1). Nếu confuse, loại/thay gesture.     |
| z-depth noise gây giảm accuracy             | Trung bình | Trung bình | Analyze confusion matrix sau T3.3; nếu z gây hại, drop → retrain 72-dim.              |
| **pynput không hoạt động trên Wayland**     | **Cao**    | **Cao**    | Test ngay trong T5.3. Fallback đã thiết kế: `ydotool` subprocess mode.                |
| Null class không đủ sample → false positive | Trung bình | Cao        | Collect Null samples trong nhiều scenario (T2.2). Nếu FP cao, dùng velocity gate.     |
| val accuracy < 88% sau training             | Thấp       | Trung bình | Collect thêm data cho confusing classes, check data quality.                          |
| Hand-leaves-frame làm mất state             | Thấp       | Thấp       | GestureStateMachine giữ state 1s qua `update_no_hand()` — đã design sẵn, test SM-7,8. |

---

## Resources Needed

- Webcam + đủ ánh sáng (data collection)
- `pynput` package
- `PyYAML` package
- Jupyter Notebook (training)
- Python 3.10+ environment
