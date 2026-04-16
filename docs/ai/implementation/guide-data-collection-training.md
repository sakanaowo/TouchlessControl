# Hướng dẫn thu thập dữ liệu & Training GRU-64

> Tài liệu hướng dẫn chi tiết cho Phase 4 — thu thập sequence data và train mô hình GRU-64.

---

## Mục lục

1. [Tổng quan](#1-tổng-quan)
2. [Chuẩn bị môi trường](#2-chuẩn-bị-môi-trường)
3. [Thu thập dữ liệu (Data Collection)](#3-thu-thập-dữ-liệu-data-collection)
4. [Kiểm tra dữ liệu](#4-kiểm-tra-dữ-liệu)
5. [Training](#5-training)
6. [Đánh giá & Validation](#6-đánh-giá--validation)
7. [Triển khai mô hình mới](#7-triển-khai-mô-hình-mới)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Tổng quan

### Pipeline

```
Thu thập → NPZ data → Training notebook → TFLite model → app.py --model gru
```

### Yêu cầu

| Metric            | Target |
| ----------------- | ------ |
| Val accuracy      | ≥ 90%  |
| Click/Drag F1     | ≥ 0.85 |
| FPS real-time     | ≥ 25   |
| Inference latency | < 35ms |

### 5 gesture classes

| Index | Tên            | Hình dạng tay                       | Hành vi          |
| ----- | -------------- | ----------------------------------- | ---------------- |
| 0     | `null`         | Tay nghỉ, chuyển động không rõ ràng | Không làm gì     |
| 1     | `pointer_move` | Chỉ ngón trỏ duỗi                   | Di chuyển chuột  |
| 2     | `left_click`   | Ngón trỏ + ngón cái chạm nhau       | Click chuột trái |
| 3     | `drag_hold`    | Nắm tay (tất cả ngón gập)           | Kéo thả (drag)   |
| 4     | `scroll_mode`  | Ngón trỏ + ngón giữa duỗi (chữ V)   | Cuộn trang       |

---

## 2. Chuẩn bị môi trường

```bash
conda activate sign
```

Kiểm tra app chạy GRU mode (sẽ dùng dummy model chưa trained):

```bash
python app.py --model gru --no-actions
```

> `--no-actions` tắt điều khiển mouse/keyboard, chỉ hiển thị camera + UI thu thập.

---

## 3. Thu thập dữ liệu (Data Collection)

### 3.1. Khởi chạy app ở chế độ thu thập sequence

```bash
python app.py --model gru --collect-mode sequence --no-actions
```

> **Mặc định**: `--model gru` tự động dùng `--collect-mode sequence`. Có thể bỏ `--collect-mode`.

```bash
python app.py --no-actions
```

### 3.2. Quy trình thu thập

1. **Mở menu class**: Nhấn `Tab`
2. **Chọn class**: Dùng `↑`/`↓` để di chuyển, `Enter` để chọn
3. **Countdown**: 3 giây chuẩn bị — giữ tay đúng pose
4. **Recording**: Màn hình viền xanh + `[REC ●]` — **giữ nguyên gesture**
5. **Done**: Hiển thị số sequences đã lưu
6. **Lặp lại**: Chọn class tiếp theo

### 3.3. Yêu cầu dữ liệu mỗi class

| Thông số                     | Giá trị                                          |
| ---------------------------- | ------------------------------------------------ |
| Số phiên (session) mỗi class | 3–5 phiên                                        |
| Thời gian mỗi phiên          | 10 giây                                          |
| Batch size khuyến nghị       | 30 (tương đương ~10s ở 30fps, mỗi 2 frame lấy 1) |
| Target sequences / class     | ≥ 50 sequences (từ sliding window stride=5)      |
| Tổng tối thiểu               | 250 sequences (50 × 5 classes)                   |

### 3.4. Góc quay và khoảng cách

Thu thập mỗi class ở **nhiều góc** để mô hình generalise:

| Biến thể                | Mô tả                 |
| ----------------------- | --------------------- |
| Trực diện               | Tay đối mặt camera    |
| Nghiêng trái 30°        | Xoay cổ tay sang trái |
| Nghiêng phải 30°        | Xoay cổ tay sang phải |
| Ngửa lên 30°            | Tay hơi ngửa          |
| Khoảng cách gần (~40cm) | Tay gần camera        |
| Khoảng cách xa (~70cm)  | Tay xa camera         |

> **Mẹo**: Mỗi phiên, thay đổi nhẹ góc và khoảng cách. KHÔNG giữ tay cứng một chỗ.

### 3.5. Lưu ý đặc biệt từng class

| Class          | Lưu ý khi thu thập                                                                         |
| -------------- | ------------------------------------------------------------------------------------------ |
| `null`         | Thu nhiều dạng: tay nghỉ, tay đang chuyển giữa pose, tay vào/ra khung hình, cầm điện thoại |
| `pointer_move` | Di chuyển tay trong khi thu thập (đừng giữ yên) — mô phỏng thao tác thực tế                |
| `left_click`   | Thực hiện nhiều kiểu pinch: chậm, nhanh, nhẹ, mạnh                                         |
| `drag_hold`    | Nắm tay chặt và di chuyển — giữ nguyên nắm tay khi kéo                                     |
| `scroll_mode`  | Giữ 2 ngón V và di chuyển lên/xuống — mô phỏng cuộn trang                                  |

### 3.6. Batch size

Điều chỉnh batch size khi menu đang mở:

- `+` / `=`: tăng 10 frames
- `-`: giảm 10 frames

### 3.7. Hủy phiên

- `Space` hoặc `ESC` khi đang countdown/recording → hủy, dữ liệu phiên đó bị xóa

---

## 4. Kiểm tra dữ liệu

Sau khi thu thập, kiểm tra NPZ file:

```python
import numpy as np

data = np.load("model/temporal_classifier/keypoint_sequences.npz")
sequences = data["sequences"]
labels = data["labels"]

print(f"Shape:   {sequences.shape}")  # (N, 20, 93)
print(f"Labels:  {labels.shape}")     # (N,)

# Phân bố class
unique, counts = np.unique(labels, return_counts=True)
CLASS_NAMES = ["null", "pointer_move", "left_click", "drag_hold", "scroll_mode"]
for cls, cnt in zip(unique, counts):
    print(f"  {CLASS_NAMES[cls]}: {cnt} sequences")
```

### Tiêu chí chấp nhận

| Tiêu chí       | Yêu cầu                                  |
| -------------- | ---------------------------------------- |
| Tổng sequences | ≥ 250                                    |
| Mỗi class      | ≥ 50 sequences                           |
| Class balance  | Class lớn nhất / class nhỏ nhất ≤ 3:1    |
| Feature range  | Giá trị nằm trong [-1, 1] (đã normalize) |

---

## 5. Training

### 5.1. Mở notebook

```bash
conda activate sign
jupyter notebook temporal_classification.ipynb
```

Hoặc mở trực tiếp trong VS Code.

### 5.2. Các bước trong notebook

| Cell | Mô tả                                                                           |
| ---- | ------------------------------------------------------------------------------- |
| 1    | Import libraries                                                                |
| 2    | Config: paths, WINDOW_SIZE=20, NUM_FEATURES=93, NUM_CLASSES=5, HIDDEN_UNITS=64  |
| 3–4  | Load NPZ & hiển thị phân bố class                                               |
| 5    | Train/Val split 75/25, stratified                                               |
| 6    | Build GRU-64: Input(20,93) → GRU(64, unroll) → Dropout(0.3) → Dense(5, softmax) |
| 7    | Train: Adam, EarlyStopping(patience=20), max 300 epochs, batch_size=128         |
| 8    | Training curves (loss + accuracy)                                               |
| 9    | Classification report + confusion matrix                                        |
| 10   | Export TFLite (dynamic quantization)                                            |
| 11   | Verify: so sánh Keras vs TFLite predictions                                     |

### 5.3. Hyperparameters

| Param      | Giá trị                         | Ghi chú                           |
| ---------- | ------------------------------- | --------------------------------- |
| Optimizer  | Adam                            | Default lr=0.001                  |
| Loss       | sparse_categorical_crossentropy |                                   |
| Epochs     | 300 (max)                       | EarlyStopping patience=20         |
| Batch size | 128                             | Giảm nếu data ít (<500 sequences) |
| GRU units  | 64                              | Không thay đổi                    |
| Dropout    | 0.3                             | Tăng lên 0.4–0.5 nếu overfit      |
| Val split  | 25%                             | Stratified                        |

### 5.4. Khi nào cần thu thêm dữ liệu

| Triệu chứng                         | Nguyên nhân                     | Giải pháp                                             |
| ----------------------------------- | ------------------------------- | ----------------------------------------------------- |
| Val acc < 80%                       | Thiếu data                      | Thu thêm 2-3 phiên mỗi class                          |
| Overfit (train>95%, val<85%)        | Data ít hoặc ít biến thể        | Thu ở nhiều góc/khoảng cách hơn, tăng Dropout         |
| F1 thấp cho 1 class cụ thể          | Class đó bị nhầm với class khác | Thu thêm class đó + xem confusion matrix              |
| `pointer_move` ↔ `scroll_mode` nhầm | Ngón giữa trùng                 | Thu cả 2 ở khoảng cách giống nhau, chú ý z-depth      |
| `left_click` ↔ `pointer_move` nhầm  | Khoảng cách ngón cái-trỏ        | left_click = tips touching; pointer_move = tips apart |

---

## 6. Đánh giá & Validation

### 6.1. Metrics cần đạt

```
              precision  recall  f1-score
null              —         —      ≥0.85
pointer_move      —         —      ≥0.85
left_click        —         —      ≥0.85
drag_hold         —         —      ≥0.85
scroll_mode       —         —      ≥0.85

accuracy                           ≥0.90
```

### 6.2. End-to-end test

Sau khi train xong, chạy app với model mới:

```bash
python app.py --model gru --no-actions
```

Kiểm tra:

| Test             | Cách kiểm                     | Pass criteria                 |
| ---------------- | ----------------------------- | ----------------------------- |
| FPS              | Đọc số FPS góc trái           | ≥ 25 FPS                      |
| Null ổn định     | Để tay nghỉ                   | Không trigger action nào      |
| Pointer smooth   | Duỗi ngón trỏ, di chuyển      | Chuột di chuyển mượt          |
| Click nhạy       | Pinch ngón trỏ + cái          | Label chuyển sang left_click  |
| Drag nhận diện   | Nắm tay + di chuyểnh          | Label chuyển sang drag_hold   |
| Scroll hoạt động | V-sign + di chuyển lên/xuống  | Label chuyển sang scroll_mode |
| Chuyển gesture   | Chuyển nhanh giữa các gesture | Không bị kẹt ở gesture cũ     |

### 6.3. Latency test

```python
import time
import numpy as np
from model.temporal_classifier import TemporalClassifier

tc = TemporalClassifier()
window = np.random.randn(20, 93).astype(np.float32)

# Warm up
for _ in range(10):
    tc(window)

# Measure
times = []
for _ in range(100):
    start = time.perf_counter()
    tc(window)
    times.append((time.perf_counter() - start) * 1000)

print(f"Avg: {np.mean(times):.2f}ms, P95: {np.percentile(times, 95):.2f}ms")
# Target: avg < 5ms, P95 < 35ms
```

---

## 7. Triển khai mô hình mới

Sau khi train thành công:

1. **Model file** đã được notebook lưu tại:
   - `model/temporal_classifier/temporal_classifier.tflite`
   - `model/temporal_classifier/temporal_classifier.keras` (backup)

2. **Chạy app** (mặc định dùng GRU):

   ```bash
   python app.py
   ```

3. **Fallback về MLP** nếu cần:

   ```bash
   python app.py --model mlp
   ```

4. **Thu thập thêm data** (dùng cùng NPZ file, data accumulate):
   ```bash
   python app.py --no-actions
   # Tab → chọn class → Enter → thu thập → train lại
   ```

---

## 8. Troubleshooting

| Vấn đề                                               | Giải pháp                                                                                                                             |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `FileNotFoundError: keypoint_sequences.npz`          | Chưa thu thập data. Chạy app với `--no-actions`, thu ít nhất 1 phiên                                                                  |
| `ValueError: need at least one array to concatenate` | NPZ rỗng hoặc bị corrupt. Xóa file, thu thập lại                                                                                      |
| FPS < 20 khi chạy GRU                                | Kiểm tra `num_threads` trong TemporalClassifier. Thử tăng lên 2                                                                       |
| App crash khi start GRU                              | Kiểm tra `model/temporal_classifier/temporal_classifier.tflite` tồn tại. Chạy `python scripts/create_dummy_temporal_model.py` nếu cần |
| Gesture bị nhầm liên tục                             | Thu thêm data cho class bị nhầm + kiểm tra confusion matrix                                                                           |
| Model quá chậm converge                              | Kiểm tra data quality: NaN, giá trị cực lớn, class balance                                                                            |
| `QT_QPA_PLATFORM` error                              | Đang chạy trên Wayland. App đã set `xcb` tự động. Nếu vẫn lỗi: `export QT_QPA_PLATFORM=xcb`                                           |

---

## Tóm tắt lệnh nhanh

```bash
# Thu thập dữ liệu (GRU mode, không điều khiển chuột)
python app.py --no-actions

# Thu thập dữ liệu (MLP mode, CSV)
python app.py --model mlp --no-actions

# Chạy app với GRU model (mặc định)
python app.py

# Chạy app với MLP fallback
python app.py --model mlp

# Chạy test suite
python -m pytest tests/ -v
```
