---
phase: planning
title: "GRU-64 Temporal Classifier — Primary Execution Plan"
description: "Chuyển đổi hệ thống từ MLP sang GRU-64 temporal sequence classifier"
feature: temporal-gesture-gru64
milestone: 2
parent: gesture-prototype-laptop-control
---

# GRU-64 Temporal Classifier — Execution Plan

> **Đây là plan chính duy nhất** của dự án. M1 (MLP prototype) đã hoàn thành và
> được giữ lại làm fallback (`--model mlp`). Tất cả công việc phát triển mới
> tập trung vào kế hoạch này.

**Design doc**: `docs/ai/design/feature-temporal-gru64.md`
**Research**: `docs/research/temporal-gesture-model-rnn.md`

---

## Milestone Overview

| Milestone | Status        | Description                                                      |
| --------- | ------------- | ---------------------------------------------------------------- |
| M0        | ✅ Done       | Research & architecture decision                                 |
| M1        | ✅ Done       | MLP prototype — 5 classes, StateMachine, ActionMapper, 111 tests |
| **M2**    | 🔄 **Active** | GRU-64 temporal classifier (this plan)                           |
| M3        | ⬜ Future     | Scaling (TD-GCN hoặc multi-gesture vocabulary)                   |

---

## Phase 1: Core Components (P0)

Tạo 4 components mới, test độc lập. Không ảnh hưởng code hiện tại.

- [x] **T1.1** — SequenceBuffer
  - File: `utils/sequence_buffer.py`
  - Test: `tests/unit/test_sequence_buffer.py`
  - Ring buffer (W, 93). push/get_window/clear/is_ready
  - Blocked by: —

- [x] **T1.2** — TemporalClassifier
  - File: `model/temporal_classifier/temporal_classifier.py`
  - Init: `model/temporal_classifier/__init__.py`
  - Label: `model/temporal_classifier/temporal_classifier_label.csv`
  - Test: `tests/unit/test_temporal_classifier.py`
  - TFLite wrapper [1,W,93]→(class_id, scores). Mock interpreter in tests
  - Blocked by: —

- [x] **T1.3** — Debouncer
  - File: `utils/debouncer.py`
  - Test: `tests/unit/test_debouncer.py`
  - Confidence threshold + cooldown + edge detect. Reuse GestureEvent
  - Blocked by: —

- [x] **T1.4** — SequenceCollector
  - File: `utils/sequence_collector.py`
  - Test: `tests/unit/test_sequence_collector.py`
  - Continuous recording → sliding window → NPZ append
  - Blocked by: —

**Gate**: All new + existing 111 tests pass ✅ (164 total)

---

## Phase 2: Training Infrastructure (P1)

- [x] **T2.1** — Create dummy TFLite model ✅ 15-04-2026
  - Script: `scripts/create_dummy_temporal_model.py`
  - Output: `model/temporal_classifier/temporal_classifier.tflite` (91.7 KB, 30853 params)
  - Keras: `model/temporal_classifier/temporal_classifier.keras`
  - TemporalClassifier loads & runs correctly (window_size=20, num_classes=5)
  - Blocked by: T1.2

- [x] **T2.2** — Training notebook ✅ 15-04-2026
  - File: `temporal_classification.ipynb`
  - 14 cells: imports, config, load NPZ, class distribution, split, build GRU-64, train, curves, confusion matrix, TFLite export, verify
  - Blocked by: T2.1

**Gate**: 172 tests pass (164 old + 8 new integration), TemporalClassifier verified with real TFLite ✅

---

## Phase 3: Integration (P2)

- [x] **T3.1** — app.py: `--model gru` flag + GRU pipeline ✅ 16-04-2026
  - Wire: SequenceBuffer → TemporalClassifier → Debouncer → ActionMapper
  - Keep: `--model mlp` fallback unchanged
  - Args: `--model gru|mlp` (default gru), `--collect-mode single|sequence` (default single)
  - GRU path: features → seq_buffer.push() → temporal_classifier() → debouncer.update() → action_mapper
  - MLP path: features → keypoint_classifier() → gesture_sm.update() → action_mapper (unchanged)
  - Blocked by: T1.1, T1.2, T1.3

- [x] **T3.2** — CollectionManager: sequence mode ✅ 16-04-2026
  - `--collect-mode sequence` → delegate to SequenceCollector
  - CollectionManager accepts optional `sequence_collector` parameter
  - Frames routed to SequenceCollector.add_frame() instead of CSV buffer
  - class_counts loaded from NPZ in sequence mode
  - Blocked by: T1.4

- [x] **T3.3** — Update package exports ✅ 16-04-2026
  - `utils/__init__.py`: +SequenceBuffer, Debouncer, SequenceCollector
  - `model/__init__.py`: +TemporalClassifier
  - Blocked by: T3.1

**Gate**: 182 tests pass (172 old + 10 new integration). All imports verified. ✅

---

## Phase 4: Data Collection & Training (P3)

**Guide**: `docs/ai/implementation/guide-data-collection-training.md`

- [ ] **T4.1** — Thu thập sequence data
  - Command: `python app.py --no-actions` (auto sequence mode)
  - 3–5 phiên × ~10s mỗi class (5 classes)
  - Target: ≥250 sequences (≥50/class), balance ratio ≤ 3:1
  - Nhiều góc: trực diện, nghiêng trái/phải 30°, gần 40cm, xa 70cm
  - Lưu ý: null thu nhiều biến thể (nghỉ, chuyển tiếp, vào/ra frame)
  - Output: `model/temporal_classifier/keypoint_sequences.npz`
  - Blocked by: T3.1, T3.2

- [ ] **T4.2** — Train GRU-64
  - Notebook: `temporal_classification.ipynb`
  - Target: val acc ≥ 90%, F1 ≥ 0.85 cho mỗi class
  - Hyperparams: Adam, batch=128, EarlyStopping patience=20, Dropout=0.3
  - Khi overfit: tăng Dropout 0.4–0.5, thu thêm biến thể
  - Khi class F1 thấp: kiểm tra confusion matrix, thu thêm class đó
  - Output: `model/temporal_classifier/temporal_classifier.tflite`
  - Blocked by: T4.1

- [ ] **T4.3** — End-to-end validation
  - `python app.py --model gru --no-actions` → kiểm tra label predictions
  - FPS ≥ 25, inference latency avg < 5ms, P95 < 35ms
  - Test: null ổn định, pointer smooth, click nhạy, drag nhận diện, scroll hoạt động
  - Blocked by: T4.2

---

## Phase 5: Cleanup (P4)

- [ ] **T5.1** — Update design docs + architecture diagram
- [ ] **T5.2** — Update AGENTS.md constraints
- [ ] **T5.3** — Deprecation markers on old modules
