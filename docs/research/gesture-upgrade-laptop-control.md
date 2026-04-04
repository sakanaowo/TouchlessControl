# Research: Nâng cấp Hand Gesture Recognition cho Laptop Control

**Date**: 2026-04-04  
**Author**: Copilot Research Agent  
**Scope**: Nhận dạng cử chỉ phức tạp đa ngón + phát hiện start/stop + điều khiển laptop

---

## Executive Summary

Hệ thống hiện tại (`app.py`) dùng MediaPipe nhưng chỉ phân loại **3 static pose** (Open/Close/Pointer) và track **duy nhất ngón trỏ** cho dynamic gesture. Đây là bottleneck chính.

**Khuyến nghị theo từng giai đoạn:**

| Giai đoạn                | Cách tiếp cận                                   | Effort     | Gain          |
| ------------------------ | ----------------------------------------------- | ---------- | ------------- |
| **Prototype (1–2 tuần)** | Feature engineering + MLP nâng cao + Null class | Thấp       | Cao           |
| **V2 (1 tháng)**         | MediaPipe + Transformer Encoder (temporal)      | Trung bình | Rất cao       |
| **Scaling**              | TD-GCN / ST-GCN trên hand skeleton              | Cao        | SOTA accuracy |

---

## 1. Task & Constraints

### Mục tiêu

- Nhận dạng **≥ 15 cử chỉ phức tạp** (multi-finger): thumbs up, V-sign, pinch, OK, gun, etc.
- Phát hiện **khi nào bắt đầu / kết thúc** một gesture (temporal segmentation)
- Ứng dụng: điều khiển laptop (volume, brightness, scroll, click, app switch...)

### Constraints

- **Hardware**: CPU laptop thông thường (không có GPU inference)
- **Latency**: ≤ 100ms end-to-end (MediaPipe ~15–30ms, còn lại cho classifier)
- **Camera**: Webcam thông thường (RGB, 30fps)
- **Scalability**: Phải dễ thêm gesture mới không cần retrain toàn bộ
- **Framework**: TFLite (hiện tại) hoặc ONNX runtime

### Phân tích hệ thống hiện tại

```
[Webcam 30fps]
    → MediaPipe Hands (21 landmarks × 2D)      ~15–30ms/frame
    → pre_process_landmark() → 42 features
    → KeyPointClassifier (MLP TFLite)           ~1ms
        Output: Open / Close / Pointer (3 classes only!)

    → point_history (16 frames × index_finger_tip)
    → PointHistoryClassifier (LSTM? TFLite)     ~2ms
        Output: Stop / Clockwise / CCW / Move (4 classes only!)
```

**Vấn đề cốt lõi:**

1. Chỉ dùng **2D** (bỏ z-depth của MediaPipe)
2. Static classifier chỉ có **3 class** — quá ít cho laptop control
3. Dynamic classifier chỉ track **1 điểm** (ngón trỏ) — không thể nhận dạng đa ngón
4. **Không có null/idle class** → model luôn phân loại, kể cả khi tay đứng yên
5. **Không có temporal segmentation** → không biết gesture đã "hoàn thành" chưa

---

## 2. SOTA Overview

### 2.1 Approaches cho Static Multi-Finger Recognition

| Approach                | Input                                       | Độ phức tạp | CPU Latency    | Accuracy            |
| ----------------------- | ------------------------------------------- | ----------- | -------------- | ------------------- |
| MLP (hiện tại)          | 42 features (2D, 21 KP)                     | Rất thấp    | <1ms           | ~70–80% (3 classes) |
| **MLP nâng cao**        | 63 features (3D) + 15 angles + 10 distances | Thấp        | ~2ms           | ~88–93%             |
| LSTM/GRU trên sequence  | 21 KP × T frames                            | Trung bình  | 5–15ms         | ~85–90%             |
| **Transformer Encoder** | 21 KP × 3D × T frames                       | Trung bình  | 10–30ms        | ~91–95%             |
| ST-GCN                  | Graph (21 nodes) × T frames                 | Cao         | 30–80ms (CPU)  | ~94–97%             |
| **TD-GCN** (SOTA 2024)  | Graph (21 nodes) × T frames                 | Cao         | 50–100ms (CPU) | ~96–98%             |

### 2.2 Approaches cho Temporal Gesture Segmentation (Start/Stop)

| Approach                  | Mô tả                                                   | Phù hợp                         |
| ------------------------- | ------------------------------------------------------- | ------------------------------- |
| **Null class**            | Thêm class "No gesture" vào classifier                  | Đơn giản nhất, prototype        |
| **Confidence threshold**  | Emit gesture khi confidence > θ cho N frames liên tiếp  | Đơn giản, hiệu quả              |
| **Velocity gating**       | Chỉ active khi wrist velocity vượt threshold            | Tự nhiên, zero-cost             |
| **HMM**                   | Hidden states: Idle → Pre-stroke → Stroke → Post-stroke | Tốt cho formal gesture protocol |
| **CTC loss**              | Continuous gesture stream, no explicit segmentation     | Cần nhiều data, complex         |
| **Sliding window + vote** | Window N frames, majority vote với hysteresis           | Dùng trong PyAutoGUI apps       |

---

## 3. Model Analysis

### 3.1 Approach A: MLP nâng cao + Null class (Prototype)

**Mô tả**: Giữ nguyên pipeline, chỉ mở rộng features và classes.

**Feature engineering:**

- `21 × 3` = 63 features (x, y, z từ MediaPipe)
- `15` joint angles (mỗi ngón 3 góc từ 3 khớp)
- `10` khoảng cách đầu ngón tới wrist + palm center
- `5` trạng thái ngón (straight/bent) dạng binary
- **Tổng: ~93 features**

**Architecture:**

```
Input(93) → Dense(256, ReLU) → Dropout(0.3)
          → Dense(128, ReLU) → Dropout(0.2)
          → Dense(N+1, Softmax)   # N gestures + 1 null
```

**Temporal segmentation**: Confidence threshold + vote window

```python
# Emit gesture chỉ khi:
# 1. confidence > 0.85
# 2. Same class xuất hiện ≥ 5/8 frames gần nhất
# 3. Khác với gesture vừa emit (debounce)
```

**Pros**: Nhanh (<2ms), dễ thêm class, không cần retrain backbone  
**Cons**: Không capture temporal pattern, dễ lẫn pose tĩnh

**Effort**: 2–3 ngày  
**Repo tham khảo**: Đây là evolution của repo hiện tại

---

### 3.2 Approach B: MediaPipe + Transformer Encoder (V2)

**Paper**: "Dynamic Hand Gesture Recognition Using MediaPipe and Transformer" (MDPI EI, 2024)  
**Key idea**: Dùng Transformer encoder để model temporal dependency trong sequence các landmark frames.

**Feature per frame:**

- `21 × 3 = 63` raw 3D landmarks (normalized về wrist)
- `21 × 3 = 63` frame-to-frame velocity (delta)
- **Total per frame: 126 features**

**Architecture:**

```
Input: [T=30 frames, 126 features]
  → Linear projection → d_model=64
  → Positional Encoding
  → Transformer Encoder (4 layers, 4 heads, FFN=128)
  → Global Average Pool over time
  → Dense(N+1, Softmax)
```

**Benchmark** (8 gesture classes, custom dataset):

- Accuracy: ~91–93%
- Params: ~140K–400K
- CPU inference on laptop: ~20–40ms cho window 30 frames

**Temporal segmentation**:

- Sliding window với stride 1 (~30ms lag)
- Class "Null/Idle" làm gate
- Hysteresis: emit gesture sau ≥ 3 frames liên tiếp cùng class

**Pros**: Captures temporal + spatial patterns, scalable, portable (ONNX)  
**Cons**: Cần collect data mới (~500 samples/class), window latency

**Effort**: 1–2 tuần (collection + training + integration)  
**Repo**: [MDPI paper 2024](https://www.mdpi.com/2673-4591/108/1/22)

---

### 3.3 Approach C: TD-GCN (SOTA — Scaling)

**Paper**: "Temporal Decoupling Graph Convolutional Network for Skeleton-Based Gesture Recognition"  
**Venue**: IEEE Transactions on Multimedia, 2024  
**Repo**: [liujf69/TD-GCN-Gesture](https://github.com/liujf69/TD-GCN-Gesture)

**Key innovation**: Graph convolution trên hand skeleton, temporal và spatial được decouple thành 2 nhánh riêng biệt để học richer patterns.

**Input**: Graph(21 nodes, 15 edges) × T frames × 3D  
**Benchmark results:**

| Dataset               | Accuracy      |
| --------------------- | ------------- |
| SHREC'17 (14 classes) | 97.5%         |
| SHREC'17 (28 classes) | 96.0%         |
| DHG-14/28             | 95.8% / 93.6% |

**CPU Inference**: ~50–100ms trên laptop (batch=1, T=32 frames) — gần giới hạn

**Pros**: SOTA accuracy, học được complex multi-finger patterns  
**Cons**: Nặng hơn Transformer, cần GPU để train (inference CPU OK), cần nhiều data hơn

**Effort**: 3–4 tuần để adapt + collect data  
**License**: MIT

---

### 3.4 Approach D: DSTSA-GCN (Mới nhất — 2025)

**Paper**: "DSTSA-GCN: Advancing Skeleton-Based Gesture Recognition with Dual-Stream Spatio-Temporal Adaptive GCN"  
**Venue**: Applied Sciences, 2025

Dual-stream: một stream dùng positional graph, một stream dùng semantic similarity graph. SOTA trên DHG-14/28.

**CPU Inference**: ~80–120ms — **có thể quá chậm** cho laptop real-time  
**Recommendation**: Chờ optimized implementation hoặc dùng cho offline analysis

---

## 4. Comparative Table

| Model               | Approach          | Params | Accuracy     | CPU Latency | Data Needed | Effort        | License  |
| ------------------- | ----------------- | ------ | ------------ | ----------- | ----------- | ------------- | -------- |
| **Current MLP**     | Static only       | ~50K   | ~70% (3 cls) | <1ms        | ✓ Exists    | —             | Apache-2 |
| **A: MLP Enhanced** | Static + features | ~200K  | ~88–92%      | ~2ms        | 100–200/cls | **2–3 days**  | -        |
| **B: Transformer**  | Temporal          | ~400K  | ~91–93%      | ~25ms       | 500/cls     | **1–2 weeks** | MIT      |
| **C: TD-GCN**       | Spatio-temporal   | ~1M    | ~96–97%      | ~70ms       | 1000+/cls   | 3–4 weeks     | MIT      |
| **D: DSTSA-GCN**    | Spatio-temporal   | ~2M    | ~97–98%      | ~100ms      | 1000+/cls   | 4–6 weeks     | -        |

**Pareto winner cho Prototype**: **Approach A** (MLP Enhanced)  
**Pareto winner cho V2**: **Approach B** (Transformer)  
**Pareto winner cho Scaling**: **Approach C** (TD-GCN, sau khi có đủ data)

---

## 5. Temporal Segmentation — Recommended Design

Vấn đề start/stop không đòi hỏi model phức tạp. Thiết kế đề xuất:

```
┌─────────────────────────────────────────────────────┐
│                  Gesture State Machine               │
│                                                      │
│  IDLE ──[motion_detected]──► TRACKING               │
│    ▲                              │                  │
│    │                    [confidence > θ, ≥N frames]  │
│    │                              ▼                  │
│    └──[confidence < θ for M frames]── ACTIVE        │
│                                   │                  │
│                           EMIT gesture event         │
└─────────────────────────────────────────────────────┘
```

**Parameters:**

- `motion_threshold`: wrist velocity > 0.02 (normalized) → bắt đầu tracking
- `confidence_threshold θ`: 0.80–0.85
- `activation_frames N`: 5 frames (~167ms @ 30fps)
- `deactivation_frames M`: 10 frames (~333ms) → kết thúc gesture
- `debounce`: không emit cùng gesture 2 lần trong 0.5s

**Null/Idle class là bắt buộc** — không có nó, model sẽ luôn emit sai gesture.

---

## 6. Feasibility Assessment

### Approach A: MLP Enhanced (Prototype)

| Question        | Assessment                                                                                                    |
| --------------- | ------------------------------------------------------------------------------------------------------------- |
| **Data**        | Cần collect ~100–200 samples/class mới. Có thể dùng data augmentation. Feasible trong 1 ngày.                 |
| **Compute**     | <2ms inference. Tổng pipeline ~20–30ms. Đủ cho 30fps. ✓                                                       |
| **Integration** | Chỉ cần sửa `pre_process_landmark()` và retrain KeyPointClassifier. Minimal changes. ✓                        |
| **Risk**        | Thiếu temporal context → dễ confuse gestures có pose tương tự. Cần gesture vocabulary được thiết kế cẩn thận. |
| **Effort**      | 2–3 ngày (1 ngày feature eng + training, 1 ngày collect data, 1 ngày integration + test)                      |

### Approach B: Transformer (V2)

| Question        | Assessment                                                                              |
| --------------- | --------------------------------------------------------------------------------------- |
| **Data**        | Cần ~500 samples/class × 15+ classes = 7500+ samples. ~2–3 ngày thu thập.               |
| **Compute**     | ~25ms inference. Tổng ~40–55ms. Đủ cho 20–25fps effective. ✓                            |
| **Integration** | Thêm temporal window buffer, thay classifier. Cần viết class mới. ~medium effort.       |
| **Risk**        | Window latency 30 frames × 33ms = ~1 giây lag. Giải quyết bằng sliding window stride=1. |
| **Effort**      | 1–2 tuần total                                                                          |

### Approach C: TD-GCN (Scaling)

| Question        | Assessment                                                                                                  |
| --------------- | ----------------------------------------------------------------------------------------------------------- |
| **Data**        | Cần 1000+ samples/class. Khó với custom laptop-control vocabulary. Consider transfer learning từ SHREC/DHG. |
| **Compute**     | ~70ms. Ổn ở 15fps nhưng gần giới hạn 100ms constraint. Có thể optimize với ONNX + int8 quantization.        |
| **Integration** | Cần viết mới graph construction từ MediaPipe landmarks. Significant re-architecture.                        |
| **Risk**        | Chưa có data đủ lớn. TD-GCN cần GPU để train hiệu quả. CPU inference có thể drop xuống <15fps.              |
| **Effort**      | 3–5 tuần + cơ sở hạ tầng training                                                                           |

---

## 7. Recommendation

### Phase 1 — Prototype (tuần 1–2): Approach A + State Machine

**Implement ngay:**

1. Mở rộng feature vector: thêm z-depth + joint angles + fingertip distances
2. Thêm `Null` class vào KeyPointClassifier
3. Mở rộng gesture vocab lên 10–15 classes (xem danh sách gợi ý bên dưới)
4. Implement Gesture State Machine với confidence voting
5. Map gesture → `pyautogui` / `pynput` actions

**Gesture vocabulary gợi ý cho laptop control:**

| Gesture         | Action                  | Mô tả                    |
| --------------- | ----------------------- | ------------------------ |
| `thumbs_up`     | Volume up / OK          | Ngón cái dựng            |
| `thumbs_down`   | Volume down             | Ngón cái xuống           |
| `open_palm`     | Pause / Play            | 5 ngón xòe               |
| `fist`          | Stop / Mute             | Nắm tay                  |
| `pointer`       | Move cursor             | Ngón trỏ dựng (hiện có)  |
| `v_sign`        | Scroll up               | V-sign (ngón trỏ + giữa) |
| `three_fingers` | Scroll down             | 3 ngón thẳng             |
| `pinch`         | Click                   | Ngón cái + trỏ chạm      |
| `spread_pinch`  | Zoom in                 | Pinch mở ra              |
| `swipe_right`   | Next slide / Switch app | Dynamic                  |
| `swipe_left`    | Prev slide / Switch app | Dynamic                  |
| `circle_cw`     | Rotate / Zoom           | Dynamic (hiện có)        |
| `circle_ccw`    | Reverse rotate          | Dynamic (hiện có)        |
| `shake`         | Undo / Close            | Wrist shake              |
| `null`          | No action               | —                        |

### Phase 2 — V2 (tháng 2): Approach B (Transformer)

Sau khi prototype chạy ổn, migrate sang Transformer để:

- Nhận dạng dynamic gestures chính xác hơn (swipe, shake)
- Phân biệt được gestures chỉ khác nhau về temporal pattern

**Architecture target:**

```python
class GestureTransformer(nn.Module):
    # Input: [batch, T=30, 126]  (21 KP × 3D + 21 KP × 3D velocity)
    # Output: [batch, N_classes]
```

Export sang ONNX runtime để inference nhanh hơn TFLite trên x86.

### Phase 3 — Scaling: TD-GCN

Khi đã có đủ data (>1000 samples/class) và cần accuracy >95%:

- Fine-tune TD-GCN pretrained trên SHREC'17
- Graph topology: adapt từ 25-joint body skeleton xuống 21-joint hand skeleton
- Deploy với ONNX int8 quantization: target ~30–40ms CPU

---

## 8. Next Steps (Immediate Actions)

- [ ] **Ngày 1**: Sửa `pre_process_landmark()` để output 3D + angles + distances
- [ ] **Ngày 1**: Thêm `null` class vào `keypoint_classifier_label.csv`
- [ ] **Ngày 2**: Thu thập data cho 10 gesture mới (~100 mỗi loại)
- [ ] **Ngày 2**: Retrain `keypoint_classifier` với new feature vector
- [ ] **Ngày 3**: Implement `GestureStateMachine` class với confidence voting
- [ ] **Ngày 3**: Map gestures → `pynput` keyboard/mouse events
- [ ] **Ngày 4–5**: Test, calibrate threshold, fix FP/FN
- [ ] **Tháng 2**: Design Transformer architecture + collect larger dataset

---

## 9. References

1. **TD-GCN** — Liu et al., "Temporal Decoupling Graph Convolutional Network for Skeleton-based Gesture Recognition", _IEEE Transactions on Multimedia_, 2024. [GitHub](https://github.com/liujf69/TD-GCN-Gesture)

2. **MediaPipe + Transformer** — "Dynamic Hand Gesture Recognition Using MediaPipe and Transformer Model for HCI", _MDPI Engineering Proceedings_, 2024. [DOI](https://www.mdpi.com/2673-4591/108/1/22)

3. **DSTSA-GCN** — "Advancing Skeleton-Based Gesture Recognition with Dual-Stream Spatio-Temporal Adaptive GCN", _Applied Sciences_, 2025.

4. **Continuous gesture survey** — "Continuous hand gesture recognition: Benchmarks and methods", _Computer Vision and Image Understanding_, 2025. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1077314225001584)

5. **MediaPipe + Feature Engineering** — Gil-Martin et al., "Hand Gesture Recognition Using MediaPipe Landmarks and Deep Learning Networks", _ICAART_, 2025. [SciTePress](https://www.scitepress.org/Papers/2025/130535/130535.pdf)

6. **Kinivi repo** (original codebase extended) — [GitHub](https://github.com/kinivi/hand-gesture-recognition-mediapipe)

7. **SHREC'17 dataset** — 14/28 hand gesture classes, skeleton-based. [Papers With Code](https://paperswithcode.com/dataset/shrec-2017-3d-shape-retrieval-contest)

8. **DHG-14/28 dataset** — [Papers With Code](https://paperswithcode.com/dataset/dhg-14-28)
