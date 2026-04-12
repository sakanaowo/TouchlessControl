---
phase: requirements
title: Guided Gesture Collection UI
description: Replace continuous-stream sampling with a structured, session-based collection workflow
---

# Requirements — Guided Gesture Collection UI

## Problem Statement

**Hệ thống thu thập mẫu gesture hiện tại có nhiều vấn đề:**

- **Ghi liên tục (continuous streaming)**: khi nhấn phím class, mỗi frame (~30fps) đều ghi 1 sample. Người dùng giơ tay 5 giây → 150 sample gần giống nhau, thiếu diversity.
- **Quên dừng**: người dùng quên nhấn Space → tiếp tục ghi sample sai class hoặc ghi lúc tay đang chuyển gesture (transition noise).
- **Chuyển class dễ nhầm**: nhấn phím khác khi chưa dừng → class cũ vẫn ghi vài frame trước khi hệ thống chuyển → data bị nhiễu.
- **Không có hướng dẫn trực quan**: không hiển thị reference pose, không biết đã thu đủ hay thiếu, không có quality feedback.
- **Phân bố mất cân bằng**: data hiện tại — class 1 (open_palm): 2433, class 5 (thumbs_down): 80. Không có cảnh báo.
- **Camera góc hẹp**: 640×480 (4:3) hạn chế vùng hiển thị tay, đặc biệt các gesture cần nhiều khoảng không (spread fingers, call_sign).

**Tham khảo hệ thống khác:**

| Hệ thống                 | Cách thu thập                                                                                                  |
| ------------------------ | -------------------------------------------------------------------------------------------------------------- |
| Google Teachable Machine | Nhấn giữ nút "Hold to Record", đếm ngược, tự dừng sau N frame, preview sample                                  |
| MediaPipe Model Maker    | Dataset từ folder ảnh (`<dataset>/<label>/*.jpg`), chạy hand detection offline, loại ảnh không detect được tay |
| Samsung Bixby Vision     | Guided pose overlay, đếm ngược 3-2-1 trước khi quét, báo "giữ yên"                                             |
| Apple ML Create          | Session-based: chọn class → capture N samples → review → confirm/discard                                       |
| HaGRID dataset           | 552K ảnh, thu từ crowd-source, chụp ảnh tĩnh (không video)                                                     |

**Pattern chung trong industry**: session-based capture với target count, countdown, auto-stop, quality gate.

## Goals & Objectives

### Primary

1. **Session-based capture**: mở class menu → chọn class → countdown 3s → capture đúng N sample → auto-stop
2. **Quality gate**: chỉ ghi sample khi hand detection confidence ≥ 0.7. Timeout 10s nếu không đủ sample → dừng + báo lỗi
3. **Visual feedback rõ ràng**: hiện class đang thu, số sample hiện tại / mục tiêu, progress bar
4. **Camera 720p (MJPG)**: tận dụng 1280×720@30fps qua MJPG codec cho góc rộng hơn
5. **Cancel session**: nhấn `Esc` hoặc `Space` khi đang countdown/recording → hủy session, discard toàn bộ sample đã thu trong session đó
6. **RAM buffer**: sample được buffer trong RAM, chỉ flush vào CSV khi session kết thúc thành công. Cancel = không flush
7. **Frame skip**: chỉ ghi 1 frame mỗi 2–3 frame để tạo diversity giữa các sample (không ghi frame liền nhau)
8. **Multi-hand**: nếu detect 2 bàn tay, ghi cả 2 (chuẩn bị cho gesture 2 tay trong tương lai)
9. **Class menu**: đọc label từ `keypoint_classifier_label.csv`, hiện danh sách class trên frame. ↑/↓ chọn, Enter bắt đầu session. Không hardcode phím cho từng class
10. **Không toggle mode**: bỏ phím `n`. Nhấn `Tab` mở/đóng class menu. Khi menu đóng = inference bình thường. Chọn class trong menu = bắt đầu session → auto-stop → tự quay lại inference

### Secondary

11. Reference pose overlay: hiện hình minh họa gesture cần thu
12. Review & discard: xem lại batch vừa thu, loại sample xấu
13. Diversity reminder: cảnh báo nếu tay giữ quá yên (sample lặp)
14. Adjustable batch size: `+`/`-` thay đổi số sample/session

### Non-goals

- Không tự augment data trong quá trình thu (augment ở bước train)
- Không thay đổi feature vector format (vẫn dùng 93-dim + 42-dim legacy)
- Không đổi cấu trúc CSV (vẫn append vào keypoint.csv)

## User Stories

1. **US-1**: Là người thu thập data, tôi muốn nhấn `Tab` → thấy danh sách class → ↑/↓ chọn → Enter → đếm ngược 3s → tự động capture 30 sample → auto-stop, để không cần nhớ dừng thủ công.
2. **US-2**: Tôi muốn thấy progress bar hiện "15/30" và class name trên màn hình, để biết đang thu gì và còn bao nhiêu.
3. **US-3**: Tôi muốn hệ thống bỏ qua frame không detect được tay hoặc confidence thấp, để tránh sample rác. Nếu sau 10s vẫn chưa đủ, hệ thống tự dừng và báo.
4. **US-4**: Tôi muốn camera có góc rộng hơn (720p) để cả bàn tay luôn nằm trong frame.
5. **US-5**: Tôi muốn hủy session bất cứ lúc nào bằng `Esc`/`Space`, và toàn bộ sample chưa hoàn thành bị discard (không ghi vào CSV).
6. **US-6**: Tôi muốn thấy cảnh báo khi data phân bố lệch (class A quá nhiều, class B quá ít).
7. **US-7**: Tôi muốn danh sách class tự động cập nhật khi sửa file label CSV, không cần sửa code.
8. **US-8**: Tôi muốn hệ thống ghi cả 2 bàn tay nếu cả 2 đều detect được, để chuẩn bị cho gesture dùng 2 tay.

## Success Criteria

| Chỉ tiêu                                | Mục tiêu                                      |
| --------------------------------------- | --------------------------------------------- |
| Không ghi sample sai class do quên dừng | 0 lần sai class per session                   |
| Thời gian onboard user mới              | < 1 phút hiểu cách dùng                       |
| Camera resolution                       | 1280×720@30fps (MJPG)                         |
| Quality gate                            | Chỉ ghi khi hand detect conf ≥ 0.7            |
| Session auto-stop                       | Dừng sau đúng N sample hoặc timeout 10s       |
| Visual progress                         | Hiện class name + count/target + progress bar |
| Cancel → discard                        | 0 sample rác ghi vào CSV khi hủy session      |
| Class menu                              | Tự đồng bộ với label CSV, scale > 13 class    |

## Constraints & Assumptions

- Webcam hỗ trợ MJPG 1280×720@30fps (đã xác nhận qua v4l2-ctl)
- MediaPipe Hands vẫn chạy được ở 720p với FPS ≥ 20
- Không thay đổi CSV format để backward-compatible
- Vẫn dùng OpenCV imshow (không web UI)
- Số lượng class không giới hạn — đọc từ label CSV, không hardcode
- Multi-hand: ghi cả 2 tay nếu detect được, mỗi tay = 1 sample row riêng

## Resolved Decisions

| #   | Câu hỏi                   | Quyết định                                                                   |
| --- | ------------------------- | ---------------------------------------------------------------------------- |
| 1   | Batch size mặc định?      | **30 sample/session** + frame skip (1 mỗi 2–3 frame) → ~2–3s recording       |
| 2   | Countdown vị trí?         | Giữa màn hình, font lớn                                                      |
| 3   | Cancel session?           | `Esc`/`Space` → discard toàn bộ buffer, không ghi CSV                        |
| 4   | Confidence thấp liên tục? | Timeout 10s → auto-stop + thông báo "Chỉ thu được M/N"                       |
| 5   | Sample diversity?         | Frame skip: ghi 1 frame mỗi 2–3 frame (không ghi liên tục)                   |
| 6   | Ghi CSV trực tiếp?        | **Buffer RAM** → flush khi session thành công. Cancel = discard              |
| 7   | Nhiều tay trong frame?    | Ghi cả 2 tay (chuẩn bị cho gesture 2 tay)                                    |
| 8   | Key mapping > 13 class?   | **Auto-menu từ label CSV**: ↑/↓ chọn, Enter start. Không hardcode phím       |
| 9   | Toggle collection mode?   | **Bỏ phím `n`**. `Tab` mở/đóng class menu. Menu đóng = inference bình thường |
| 10  | Đổi batch size?           | `+`/`-` khi menu mở (secondary goal)                                         |

## Questions & Open Items

1. Sound feedback (beep) khi capture xong? → Secondary iteration
2. Review mode: xem sample thumbnail? → Secondary goal, không bắt buộc M1
