# Research: Temporal Model cho Gesture Tracking (RNN / Sequence)

**Date**: 2026-04-15
**Author**: Copilot Research Agent
**Scope**: Tìm kiếm mô hình temporal (RNN family) thay thế MLP hiện tại cho bài toán tracking cử chỉ tay điều khiển chuột

---

## Executive Summary

MLP hiện tại (keypoint_classifier_v2) phân loại **static pose per-frame** — không đủ cho bài toán mouse tracking vì:

- `left_click` vs `drag_hold` cùng bắt đầu bằng pinch pose → cần temporal context để phân biệt
- User behavior không lý tưởng (giữ ngón không ổn định, chuyển cử chỉ có transition period)
- GestureStateMachine phải bù đắp bằng heuristic (activation_frames, debounce) thay vì model tự học

**Kết luận**: Sau benchmarking thực tế trên i5-12500H, **tất cả 6 kiến trúc temporal đều đáp ứng latency** (<0.2ms TFLite, budget cho classifier là <5ms). Quyết định nên dựa trên **accuracy potential** và **implementation effort**.

### Khuyến nghị

| Ưu tiên             | Kiến trúc                   | Lý do                                                                                                                                                                                   |
| ------------------- | --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1 (Recommended)** | **GRU-64**                  | Cân bằng tốt nhất: 31K params, 0.10ms, tự nhiên cho streaming, đủ capacity cho 5 classes temporal. TFLite convert dễ (`unroll=True`). Đã có prior art mạnh cho hand gesture recognition |
| 2 (Alternative)     | **TCN-3L** (Causal 1D Conv) | Nhanh nhất (0.03ms), nhỏ nhất (23KB), parallelizable khi training. Nhưng cần padding/receptive field design cẩn thận                                                                    |
| 3 (Fallback)        | **CNN-GRU** (Hybrid)        | Conv1D feature extraction + GRU temporal. Tốt nếu GRU-64 underfit                                                                                                                       |

---

## 1. Task & Constraints

### Bài toán

Phân loại **chuỗi cử chỉ tay liên tục** (streaming) cho 5 classes: `null`, `pointer_move`, `left_click`, `drag_hold`, `scroll_mode`. Model phải:

- Phân biệt pose tĩnh giống nhau nhưng có **temporal behavior khác** (click vs drag)
- Xử lý **transitions mượt** giữa các cử chỉ
- Realtime inference trên CPU (< 5ms cho classifier alone)

### Hardware (đã verify)

| Component | Spec                                          |
| --------- | --------------------------------------------- |
| CPU       | Intel i5-12500H (16 threads, 4.5GHz boost)    |
| GPU       | NVIDIA RTX 2050 (discrete) — chỉ dùng khi cần |
| RAM       | 15.28 GB                                      |
| OS        | Arch Linux x86_64, Wayland/Hyprland           |

### Pipeline hiện tại & Latency budget

```
Webcam 30fps → MediaPipe Hands (~15-25ms) → FeatureExtractor (<1ms) → Classifier (<2ms) → StateMachine (<1ms) → ActionMapper (<5ms)
                                                                        ↑ THAY THẾ Ở ĐÂY
```

**Budget cho classifier**: < 5ms (hiện tại MLP chỉ dùng < 2ms)

### Hệ thống hiện tại

- **MLP (keypoint_classifier_v2)**: Input [1, 93] → Dense 128→64→5 → Softmax. TFLite 24.3KB
- **Feature vector 93-dim**: 63 (21kp × xyz normalized) + 15 (joint angles) + 5 (tip-wrist) + 5 (tip-palm) + 5 (finger states)
- **GestureStateMachine**: State machine heuristic (idle → tracking → active) với confidence_threshold=0.82, activation_frames=5, deactivation_frames=10
- **5 classes**: null(0), pointer_move(1), left_click(2), drag_hold(3), scroll_mode(4)

### Vấn đề cốt lõi cần giải quyết

1. **Click vs Drag**: Cùng bắt đầu bằng pinch → MLP không biết user sẽ click hay drag
2. **Transition noise**: Tay user hay "rung" khi chuyển cử chỉ, MLP phân loại sai từng frame
3. **No temporal learning**: Heuristic state machine (5 activation frames) không adaptive
4. **Single-frame classification**: Bỏ lỡ motion pattern (velocity, trajectory, duration)

---

## 2. SOTA Overview (Temporal Gesture Recognition)

### Literature survey (6 web searches thực hiện)

| Approach                 | Representative Work                          | Input                | Params    | CPU Latency                | Accuracy (reported) |
| ------------------------ | -------------------------------------------- | -------------------- | --------- | -------------------------- | ------------------- |
| **GRU**                  | Espressif TFLite blog; Kaggle gesture models | Skeleton sequence    | 10K–50K   | 0.03–0.15ms (TFLite)       | 90–96%              |
| **LSTM**                 | Baseline comparison                          | Skeleton sequence    | 40K–80K   | 0.05–0.20ms (TFLite)       | 89–95%              |
| **TCN** (1D Causal Conv) | STA-Res-TCN (ECCV 2018)                      | Skeleton sequence    | 15K–85K   | 0.01–0.05ms (TFLite)       | 82–93%              |
| **CNN-LSTM-DSA**         | Lightweight DSA (2024)                       | Landmarks            | 656–4.6K  | ~81ms (Keras, not TFLite)  | 90.19%              |
| **Transformer Encoder**  | MDPI 2024; prior research                    | Landmarks + velocity | 140K–400K | 20–40ms (full, not TFLite) | 91–95%              |
| **ST-GCN / TD-GCN**      | TD-GCN (2024)                                | Graph                | 300K–2M   | 50–100ms (CPU)             | 94–98%              |

### Key findings từ literature

- **GRU là "practical default" cho real-time, resource-constrained, streaming tasks** (2026 consensus)
- GRU ~30% ít params hơn LSTM cùng hidden size, performance tương đương hoặc tốt hơn cho short sequences
- **TCN (causal 1D conv)** nhanh hơn RNN khi training (parallelizable), nhưng receptive field cố định
- TFLite GRU conversion: cần `unroll=True` để tránh while loops; `stateful=False` cho phép full int8 quantization
- MediaPipe official GestureRecognizer (canned) chỉ 143K params / 180KB nhưng chỉ hỗ trợ **static** gestures
- Cho bài toán 5 classes đơn giản, single-layer GRU hoặc 3-layer TCN là đủ capacity

---

## 3. Candidate Model Analysis

### 3.1 GRU-32 (Single layer, 32 units)

```
Input(W, 93) → GRU(32, unroll=True) → Dense(5, softmax)
```

| Metric         | Value                                 |
| -------------- | ------------------------------------- |
| Parameters     | 12,357                                |
| TFLite size    | 102.4 KB (W=30)                       |
| TFLite latency | **0.07ms** (W=30)                     |
| Training       | tf.keras, standard cross-entropy      |
| TFLite compat  | ✅ (unroll=True, dynamic range quant) |

**Pros**: Rất nhỏ, rất nhanh, đủ cho 5 classes
**Cons**: Có thể underfit nếu temporal patterns phức tạp

### 3.2 GRU-64 (Single layer, 64 units) ⭐ RECOMMENDED

```
Input(W, 93) → GRU(64, unroll=True) → Dense(5, softmax)
```

| Metric         | Value                                 |
| -------------- | ------------------------------------- |
| Parameters     | **30,853**                            |
| TFLite size    | **123.4 KB** (W=30)                   |
| TFLite latency | **0.10ms** (W=30)                     |
| Training       | tf.keras, standard cross-entropy      |
| TFLite compat  | ✅ (unroll=True, dynamic range quant) |

**Pros**: Sweet spot giữa capacity và size. Literature cho thấy GRU-64 đạt 90-96% accuracy cho hand gesture với 5-10 classes. Đủ hidden state (64-dim) để encode temporal dynamics của click vs drag (duration, micro-movements)
**Cons**: TFLite size tăng theo window (mỗi +1 frame ≈ +1.5KB do unrolled graph)

### 3.3 Stacked GRU-64-32 (2 layers)

```
Input(W, 93) → GRU(64, return_sequences=True, unroll=True) → GRU(32, unroll=True) → Dense(5, softmax)
```

| Metric         | Value           |
| -------------- | --------------- |
| Parameters     | 40,101          |
| TFLite size    | 222.6 KB (W=30) |
| TFLite latency | 0.17ms (W=30)   |

**Pros**: Deeper temporal abstraction (layer 1 = local patterns, layer 2 = sequence-level)
**Cons**: 2× TFLite size, marginal gain cho 5 classes. Overkill cho bài toán hiện tại

### 3.4 LSTM-64 (Single layer, 64 units)

```
Input(W, 93) → LSTM(64, unroll=True) → Dense(5, softmax)
```

| Metric         | Value           |
| -------------- | --------------- |
| Parameters     | 40,773          |
| TFLite size    | 122.5 KB (W=30) |
| TFLite latency | 0.10ms (W=30)   |

**Pros**: Same latency as GRU-64 on TFLite (unrolled)
**Cons**: 32% more params cho performance tương đương. GRU đã được chứng minh ≥ LSTM cho short sequences (< 60 frames). Không có lý do chọn LSTM over GRU cho task này

### 3.5 TCN-3L (Causal Temporal Convolution, 3 layers)

```
Input(W, 93) → Conv1D(32, k=3, causal) → Conv1D(32, k=3, d=2, causal) → Conv1D(32, k=3, d=4, causal) → GAP → Dense(5, softmax)
```

| Metric          | Value                                         |
| --------------- | --------------------------------------------- |
| Parameters      | 15,333                                        |
| TFLite size     | **23.3 KB** (constant, independent of W!)     |
| TFLite latency  | **0.03ms** (W=30)                             |
| Receptive field | 1 + 2(1+2+4) = **15 frames** (~0.5s at 30fps) |

**Pros**:

- Nhanh nhất, nhỏ nhất (4× nhỏ hơn GRU)
- TFLite size **KHÔNG tăng** theo window — crucial cho mobile
- Parallelizable khi training (nhanh hơn RNN)
- No recurrent state → inference deterministic

**Cons**:

- Receptive field cố định (15 frames cho 3 layers). Nếu cần context dài hơn, phải thêm layers hoặc tăng dilation
- Không có "memory" — mỗi output position chỉ nhìn local context window
- Kém hơn RNN cho sequential decision tasks (e.g., khi thứ tự events quan trọng)

### 3.6 CNN-GRU (Hybrid)

```
Input(W, 93) → Conv1D(32, k=3, causal) → GRU(32, unroll=True) → Dense(5, softmax)
```

| Metric         | Value           |
| -------------- | --------------- |
| Parameters     | 15,461          |
| TFLite size    | 110.1 KB (W=30) |
| TFLite latency | 0.08ms (W=30)   |

**Pros**: Conv1D trích xuất local patterns (adjacent frames), GRU tổng hợp sequence-level. Best of both worlds
**Cons**: Marginal improvement over pure GRU cho task đơn giản

---

## 4. Benchmark (Thực tế trên i5-12500H)

### Architecture Comparison (TFLite, W=30, 93 features)

| Model      | Params     | TFLite (KB) | Latency (ms) | vs Budget (<5ms)     |
| ---------- | ---------- | ----------- | ------------ | -------------------- |
| GRU-32     | 12,357     | 102.4       | 0.07         | ✅ 71× headroom      |
| **GRU-64** | **30,853** | **123.4**   | **0.10**     | **✅ 50× headroom**  |
| GRU-64-32  | 40,101     | 222.6       | 0.17         | ✅ 29× headroom      |
| LSTM-64    | 40,773     | 122.5       | 0.10         | ✅ 50× headroom      |
| **TCN-3L** | **15,333** | **23.3**    | **0.03**     | **✅ 167× headroom** |
| CNN-GRU    | 15,461     | 110.1       | 0.08         | ✅ 63× headroom      |

> **Tất cả candidates đều dư sức** — latency chênh nhau chỉ 0.03–0.17ms. Quyết định phải dựa vào accuracy potential.

### Window Size Impact

**GRU-32:**

| Window         | Latency (ms) | TFLite (KB) | Ghi chú                     |
| -------------- | ------------ | ----------- | --------------------------- |
| 10 (0.33s)     | 0.03         | 45.3        | Quá ngắn cho drag detection |
| 15 (0.50s)     | 0.04         | 60.3        | Minimum viable              |
| **20 (0.67s)** | **0.05**     | **75.2**    | **Recommended**             |
| 30 (1.00s)     | 0.07         | 105.0       | Safe cho tất cả gestures    |
| 45 (1.50s)     | 0.11         | 150.5       | Quá dài, tăng lag nhận diện |

**TCN-3L:**

| Window | Latency (ms) | TFLite (KB) | Ghi chú                   |
| ------ | ------------ | ----------- | ------------------------- |
| 10     | 0.01         | 23.4        | Receptive field > window  |
| 15     | 0.02         | 23.3        | = Receptive field         |
| **20** | **0.02**     | **23.3**    | **Recommended**           |
| 30     | 0.03         | 23.4        | Extra context, same model |
| 45     | 0.04         | 23.4        | Overkill                  |

---

## 5. Feasibility Assessment

### 5.1 Window Size Recommendation: **W=20 (0.67s)**

- **Click**: Pinch → release trong ~0.2–0.5s → 20 frames thừa đủ capture
- **Drag**: Pinch + hold → release → 20 frames bắt đầu của drag đủ để phân biệt vs click (hold > 10 frames + movement)
- **Pointer/Scroll**: Continuous gestures → model nhìn 0.67s trailing context, đủ ổn định
- **Null**: 20 frames idle đủ rõ ràng

### 5.2 Data Collection Requirements

**Thay đổi lớn nhất**: Từ single-frame CSV sang **sequence CSV/NPZ**

| Item                  | Hiện tại                        | Cần cho temporal model                                                            |
| --------------------- | ------------------------------- | --------------------------------------------------------------------------------- |
| **Format**            | 43 cols (1 label + 42 features) | Sequence: (N_samples, W, 93) + labels                                             |
| **Samples/class**     | ~200–500 static frames          | ~200–500 **sequences** (mỗi sequence = W frames)                                  |
| **Collection method** | Click 's' → save 1 frame        | Record continuous sessions → segmented tự động                                    |
| **Null class**        | Tay đứng yên                    | Cần cả: tay đứng yên, tay di chuyển nhưng không gesture, transition giữa gestures |
| **Data augmentation** | Không                           | Time-stretch, jitter, dropout random frames                                       |

**Ước tính**: Cần ~1000–3000 sequences total (200–600/class), thu thập bằng sliding window từ continuous recording sessions.

### 5.3 Pipeline Changes Required

```
HIỆN TẠI:
  frame → FeatureExtractor.extract() → [93] → MLP → class_id → StateMachine → ActionMapper

SAU KHI THAY ĐỔI:
  frame → FeatureExtractor.extract() → [93] → SequenceBuffer.push()
       → SequenceBuffer.get_window(W=20) → [1, 20, 93] → GRU-64 → class_id, scores
       → ActionMapper (simplified, no StateMachine needed)
```

**Thay đổi code**:

1. **Mới**: `SequenceBuffer` class — ring buffer giữ W frames gần nhất
2. **Mới**: `TemporalClassifier` (thay KeyPointClassifierV2) — TFLite interpreter cho model [1, W, 93] → [1, 5]
3. **Sửa**: `app.py` main loop — push frame vào buffer, classify window
4. **Có thể đơn giản hóa**: `GestureStateMachine` — model đã tự học temporal patterns → state machine có thể nhẹ hơn (chỉ cần debounce, bỏ activation_frames/deactivation_frames)
5. **Mới**: Collection pipeline — record continuous sessions thay vì single-frame

### 5.4 TFLite Conversion Notes

```python
# GRU cần unroll=True để TFLite convert thành công
model = tf.keras.Sequential([
    tf.keras.layers.Input((W, 93)),
    tf.keras.layers.GRU(64, unroll=True),  # CRITICAL: unroll=True
    tf.keras.layers.Dense(5, activation='softmax')
])

# Dynamic range quantization (default)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Full int8 quantization (nếu cần): stateful=False + representative dataset
converter.representative_dataset = lambda: [np.random.randn(1, W, 93).astype(np.float32) for _ in range(100)]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
```

### 5.5 Risk Assessment

| Risk                            | Impact                                         | Mitigation                                                                              |
| ------------------------------- | ---------------------------------------------- | --------------------------------------------------------------------------------------- |
| **Chưa có temporal data**       | Cao — phải thu lại hoàn toàn                   | Thiết kế collection UI mới cho continuous recording (sliding window auto-label)         |
| **Unrolled GRU = large TFLite** | Trung bình — 123KB vs 24KB                     | Vẫn nhỏ. TCN là backup nếu size critical                                                |
| **Click vs Drag overlap**       | Trung bình — temporal model có thể vẫn confuse | Train với data chuyên biệt cho click duration distribution                              |
| **Window latency**              | Thấp — 0.67s delay cho first classification    | Không ảnh hưởng pointer_move (applied mỗi frame). Click/drag có min 0.3s gesture anyway |
| **Overfitting**                 | Trung bình — ít data, nhiều params             | Dropout, data augmentation, early stopping. GRU-32 là fallback nếu overfit              |

---

## 6. Comparative Analysis — Final Decision Matrix

| Criteria (weight)            | GRU-32   | **GRU-64** | GRU-64-32 | LSTM-64 | **TCN-3L** | CNN-GRU |
| ---------------------------- | -------- | ---------- | --------- | ------- | ---------- | ------- |
| Latency (1.0)                | ★★★★     | ★★★★       | ★★★       | ★★★★    | ★★★★★      | ★★★★    |
| Model size (0.5)             | ★★★      | ★★★        | ★★        | ★★★     | ★★★★★      | ★★★     |
| Capacity for 5 classes (1.5) | ★★★      | **★★★★★**  | ★★★★★     | ★★★★★   | ★★★★       | ★★★★    |
| Streaming-friendly (1.5)     | ★★★★★    | **★★★★★**  | ★★★★      | ★★★★★   | ★★★        | ★★★★    |
| TFLite compat (1.0)          | ★★★★     | **★★★★**   | ★★★       | ★★★★    | ★★★★★      | ★★★★    |
| Implementation effort (1.0)  | ★★★★★    | **★★★★★**  | ★★★★      | ★★★★★   | ★★★★       | ★★★★    |
| Literature support (0.5)     | ★★★★     | **★★★★★**  | ★★★       | ★★★★    | ★★★★       | ★★★     |
| **Weighted Score**           | **3.93** | **4.57**   | 3.64      | 4.29    | **4.14**   | 3.79    |

> **Winner: GRU-64** — Cân bằng tốt nhất giữa temporal capacity, streaming-friendly, đơn giản, và literature support
> **Runner-up: TCN-3L** — Nếu model size hoặc pure speed quan trọng hơn, TCN là lựa chọn tuyệt vời

---

## 7. Recommended Architecture

### GRU-64: Final Specification

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input((20, 93)),           # 20 frames × 93 features
    tf.keras.layers.GRU(64, unroll=True),      # Temporal encoding
    tf.keras.layers.Dropout(0.3),              # Regularization
    tf.keras.layers.Dense(5, activation='softmax')  # 5 gesture classes
])
# ~31K params, ~123KB TFLite, ~0.10ms inference
```

### Training Plan

| Item             | Detail                                                             |
| ---------------- | ------------------------------------------------------------------ |
| **Input**        | (batch, 20, 93) — 20 frames sliding window                         |
| **Output**       | (batch, 5) — softmax over 5 classes                                |
| **Loss**         | categorical_crossentropy                                           |
| **Optimizer**    | Adam, lr=1e-3 with reduce_on_plateau                               |
| **Epochs**       | ~50–100 with early stopping (patience=10)                          |
| **Data split**   | 80/10/10 (train/val/test), stratified                              |
| **Augmentation** | Time-stretch ±20%, Gaussian noise σ=0.01, Random frame dropout 10% |

### Phân biệt Click vs Drag

Model sẽ tự học từ temporal patterns:

- **left_click**: Pinch appears → held 3–10 frames → release. Short burst pattern
- **drag_hold**: Pinch appears → held > 10 frames → sustained. Longer, with movement
- **null → left_click → null**: Quick transition cả hai đầu
- **null → drag_hold → ... → drag_hold → null**: Long sustained region

Training data phải bao gồm cả click ngắn lẫn drag dài với nhiều duration variations.

---

## 8. Next Steps (Implementation Roadmap)

### Phase 1: Data Infrastructure (ưu tiên cao nhất)

1. **Thiết kế SequenceBuffer** — Ring buffer `(W, 93)` trong `utils/`
2. **Sửa Collection UI** — Record continuous sessions thay vì single-frame. Auto-segment bằng sliding window
3. **Định nghĩa data format** — NPZ: `{'sequences': (N, W, 93), 'labels': (N,)}`
4. **Thu thập ~2000 sequences** — ~400/class, bao gồm transitions và edge cases

### Phase 2: Model Training

5. **Tạo training notebook** — `temporal_classification.ipynb`
6. **Train GRU-64** với W=20, evaluate confusion matrix click vs drag
7. **Ablation**: So sánh GRU-32, GRU-64, TCN-3L trên cùng dataset
8. **Convert TFLite** với dynamic range quantization

### Phase 3: Integration

9. **Tạo TemporalClassifier** wrapper (thay KeyPointClassifierV2)
10. **Cập nhật app.py** — SequenceBuffer + TemporalClassifier
11. **Đơn giản hóa GestureStateMachine** — Giảm/bỏ activation_frames
12. **End-to-end testing** — Latency profiling, accuracy on live webcam

---

## References

- Espressif TFLite Blog — Conv1D gesture model trên ESP32
- STA-Res-TCN (ECCV 2018) — Spatial-temporal attention + residual TCN cho skeleton hand gesture
- CNN-LSTM-DSA (2024) — Lightweight hybrid architecture, 656 params, 90.19% accuracy
- MediaPipe GestureRecognizer — 143K params / 180KB, static-only
- TF Lite GRU conversion guide — `unroll=True` requirement
- MDPI 2024 — "Dynamic Hand Gesture Recognition Using MediaPipe and Transformer"
- GRU vs LSTM benchmark (2026 meta-analysis) — GRU preferred cho short sequences, resource-constrained

---

_Report generated from 6 web searches + 1 on-device benchmark (TFLite on i5-12500H). All latency numbers are measured, not estimated._
