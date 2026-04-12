Optimizing tool selection...# Báo cáo Check-Implementation (Bản tiếng Việt)

## Phạm vi

Đối chiếu phần triển khai tính năng **Giao diện Thu thập Cử chỉ Có Hướng dẫn** với:

- Tài liệu thiết kế
- Tài liệu yêu cầu

Các phần đã rà soát: mô-đun menu lớp, mô-đun quản lý thu thập, và phần vòng lặp/chồng lớp hiển thị trong ứng dụng chính.

---

## Tóm tắt

| Hạng mục                                | Số lượng |
| --------------------------------------- | -------- |
| ✅ Điểm yêu cầu/thiết kế đáp ứng đầy đủ | 38       |
| ⚠️ Sai khác hiển thị nhỏ                | 5        |
| 🔶 Sai khác hành vi                     | 2        |
| ❌ Tính năng còn thiếu                  | 0        |

**Đánh giá tổng quan:** Triển khai tốt. Các phần cốt lõi như FSM, toàn vẹn dữ liệu, quality gate, hủy/timeout, multi-hand, pipeline CSV, menu lớp và xử lý phím đều bám sát đặc tả.

---

## So sánh theo thành phần

### 1) Menu lớp

| Đặc tả thiết kế                              | Trạng thái | Ghi chú                                |
| -------------------------------------------- | ---------- | -------------------------------------- |
| Đọc nhãn từ CSV, hot-reload khi bật/tắt menu | ✅         | Có gọi hàm reload khi toggle           |
| Điều hướng ↑/↓ có vòng lặp                   | ✅         |                                        |
| Xác nhận trả về class_id                     | ✅         | Trả về số nguyên hoặc None khi ẩn/rỗng |
| Overlay bán trong suốt, có highlight         | ✅         |                                        |
| Hiển thị số mẫu theo lớp                     | ✅         |                                        |
| Giới hạn lại chỉ số khi số nhãn giảm         | ✅         |                                        |

**Nhỏ:** Thiết kế nêu hàm public `reload_labels()`, triển khai dùng hàm private `_reload_labels()`. Về chức năng là tương đương.

---

### 2) Quản lý thu thập

| Đặc tả thiết kế                                              | Trạng thái | Ghi chú                       |
| ------------------------------------------------------------ | ---------- | ----------------------------- |
| FSM: idle → countdown → recording → done → idle              | ✅         |                               |
| Đệm RAM, chỉ flush khi thành công                            | ✅         |                               |
| Cancel = bỏ toàn bộ buffer                                   | ✅         |                               |
| Timeout = lưu phần đã thu                                    | ✅         |                               |
| Quality gate ≥ 0.7 (loại nếu tất cả tay đều thấp hơn ngưỡng) | ✅         |                               |
| Frame skip mặc định = 2                                      | ✅         |                               |
| Multi-hand: 1 frame tăng collected 1, buffer tăng N dòng     | ✅         |                               |
| `collected` đếm theo frame, không đếm theo row               | ✅         |                               |
| Tự đếm CSV hiện có khi khởi tạo                              | ✅         |                               |
| `class_counts` theo số row CSV                               | ✅         |                               |
| Bắt đầu session sẽ reset buffer + bộ đếm                     | ✅         |                               |
| Dữ liệu tay gồm đặc trưng + độ tin cậy                       | ✅         | Khác tên trường, cùng ý nghĩa |

Sai khác:

- ⚠️ `DONE_DISPLAY_SECONDS = 1.5` trong khi thiết kế ghi 1.0s.
- ⚠️ Docstring của `on_frame` ghi “0, 1, hoặc 2” nhưng thực tế trả về 0 hoặc 1 (đúng theo luật đếm frame).

---

### 3) Xử lý phím trong ứng dụng chính

| Đặc tả thiết kế                                   | Trạng thái | Ghi chú                         |
| ------------------------------------------------- | ---------- | ------------------------------- |
| Tab bật/tắt menu (chỉ khi idle)                   | ✅         | Có chặn khi countdown/recording |
| ↑/↓ điều hướng menu                               | ✅         |                                 |
| Enter xác nhận + bắt đầu session                  | ✅         |                                 |
| Ưu tiên Esc: hủy session > đóng menu > thoát      | ✅         |                                 |
| Space để hủy session                              | ✅         |                                 |
| Bỏ phím `n`                                       | ✅         | Đã thay bằng luồng Tab/session  |
| Dùng `waitKeyEx` cho phím mũi tên                 | ✅         |                                 |
| `max_num_hands=2`                                 | ✅         |                                 |
| MJPG 720p (set FOURCC trước độ phân giải)         | ✅         |                                 |
| Cảnh báo khi độ phân giải không đạt               | ✅         |                                 |
| Khóa class trong lúc recording (toàn vẹn dữ liệu) | ✅         |                                 |
| Gọi `tick()` mỗi frame                            | ✅         |                                 |
| Đồng bộ `class_counts` vào menu mỗi frame         | ✅         |                                 |
| Chế độ point-history qua phím `h`                 | ✅         |                                 |
| Chỉ feed `on_frame` khi countdown/recording       | ✅         |                                 |

Sai khác hành vi:

- 🔶 Phím `+/-` chỉnh batch size đang dùng được ở mọi trạng thái.
- Yêu cầu gốc: chỉ cho chỉnh khi menu đang mở.

---

### 4) Overlay hiển thị

| Đặc tả thiết kế                                    | Trạng thái | Ghi chú |
| -------------------------------------------------- | ---------- | ------- |
| Countdown số lớn ở giữa                            | ✅         |         |
| Countdown hiển thị tên class bên dưới              | ✅         |         |
| Recording có viền xanh                             | ✅         |         |
| Hiển thị `[REC] label count/target` + progress bar | ✅         |         |
| Done hiển thị thông điệp success/timeout           | ✅         |         |
| Biểu đồ cân bằng theo thanh ngang + số lượng       | ✅         |         |
| Mã màu (xanh ≥100, cam nếu thấp hơn)               | ✅         |         |
| Info bar: FPS + gợi ý khi idle                     | ✅         |         |

Sai khác hiển thị:

- ⚠️ Thiếu dòng “Giữ tay đúng pose!” khi countdown.
- ⚠️ Thiếu dấu `●` trong chỉ báo `[REC ●]`.
- ⚠️ Thiếu tiền tố `✓`/`⚠` trong thông báo done.
- ⚠️ Thiếu cụm “for {class_name}” trong thông báo thành công.
- ⚠️ Biểu đồ cân bằng chưa có chữ “NEEDS MORE” cho lớp còn ít mẫu.

---

## Khuyến nghị xử lý sai khác

1. Ràng buộc chỉnh batch size theo trạng thái menu (mức ưu tiên: cao vừa)  
   Hiện tại `+/-` hoạt động mọi lúc; nên thêm guard chỉ cho phép khi menu mở.

2. Thống nhất thời gian hiển thị DONE (mức ưu tiên: thấp)  
   Giữ 1.5s hoặc đổi về 1.0s, nhưng cần thống nhất giữa code và tài liệu.

3. Đợt polish giao diện (mức ưu tiên: thấp)  
   Bổ sung các chi tiết chữ/icon còn thiếu ở countdown, REC, done và balance chart.

---

## Độ bao phủ kiểm thử

- Tổng hiện có: **82 test**, tất cả pass.
- Bao gồm unit test cho menu lớp, quản lý thu thập, gesture state machine, feature extractor.
- Có thêm integration test cho luồng đầy đủ: chọn lớp → countdown → recording → ghi CSV, cùng các ca cancel, timeout, hot-reload, multi-hand, frame skip.

Các khoảng trống cần test thủ công:

1. Độ đúng hiển thị overlay (so sánh trực quan/screenshot).
2. Timing phím thực tế với webcam và `waitKeyEx`.
3. Xử lý lỗi ghi CSV khi hết dung lượng đĩa.
4. Fallback camera khi MJPG 720p không khả dụng.

---

## Kết luận

Phần triển khai bám sát thiết kế và yêu cầu. Hai sai khác hành vi đều rủi ro thấp và sửa nhanh. Nhóm sai khác còn lại chủ yếu là polish giao diện. Các cam kết quan trọng về toàn vẹn dữ liệu, cancel/timeout, quality gate, frame skip và multi-hand đã được xác nhận qua bộ test pass toàn bộ.
