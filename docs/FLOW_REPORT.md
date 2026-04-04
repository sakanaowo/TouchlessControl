# Báo Cáo Chi Tiết Luồng Hoạt Động

# Hand Gesture Recognition using MediaPipe

---

## Mục Lục

1. [Tổng Quan Kiến Trúc Hệ Thống](#1-tổng-quan-kiến-trúc-hệ-thống)
2. [Luồng Khởi Động Ứng Dụng](#2-luồng-khởi-động-ứng-dụng)
3. [Vòng Lặp Xử Lý Chính](#3-vòng-lặp-xử-lý-chính)
4. [Luồng Phát Hiện Tay & Trích Xuất Landmark](#4-luồng-phát-hiện-tay--trích-xuất-landmark)
5. [Luồng Tiền Xử Lý Dữ Liệu](#5-luồng-tiền-xử-lý-dữ-liệu)
6. [Luồng Phân Loại Cử Chỉ Tĩnh (KeyPoint Classifier)](#6-luồng-phân-loại-cử-chỉ-tĩnh-keypoint-classifier)
7. [Luồng Phân Loại Cử Chỉ Động (Point History Classifier)](#7-luồng-phân-loại-cử-chỉ-động-point-history-classifier)
8. [Luồng Quản Lý Chế Độ & Ghi Dữ Liệu Huấn Luyện](#8-luồng-quản-lý-chế-độ--ghi-dữ-liệu-huấn-luyện)
9. [Luồng Kết Xuất Hình Ảnh](#9-luồng-kết-xuất-hình-ảnh)
10. [Luồng Xử Lý Toàn Bộ Một Frame](#10-luồng-xử-lý-toàn-bộ-một-frame)
11. [Sơ Đồ Thành Phần Hệ Thống](#11-sơ-đồ-thành-phần-hệ-thống)
12. [Sơ Đồ Trạng Thái Chế Độ](#12-sơ-đồ-trạng-thái-chế-độ)
13. [Luồng Pipeline Mô Hình TFLite](#13-luồng-pipeline-mô-hình-tflite)

---

## 1. Tổng Quan Kiến Trúc Hệ Thống

Hệ thống nhận diện cử chỉ tay gồm 3 tầng chính:

- **Tầng Thu Thập** — Camera → OpenCV capture
- **Tầng Nhận Diện** — MediaPipe Hands → 21 landmark điểm tay
- **Tầng Phân Loại** — Hai mô hình TFLite song song: cử chỉ tĩnh + cử chỉ động

```mermaid
graph TB
    subgraph InputLayer["🎥 Tầng Thu Thập"]
        CAM[Camera / Webcam]
        CV[OpenCV VideoCapture]
    end

    subgraph DetectionLayer["🤚 Tầng Nhận Diện - MediaPipe"]
        MP[MediaPipe Hands]
        LM["21 Landmarks\n(x, y mỗi điểm)"]
        BRECT[Bounding Rectangle]
    end

    subgraph PreprocessLayer["⚙️ Tầng Tiền Xử Lý"]
        PP1["pre_process_landmark()\n→ tọa độ tương đối + normalize"]
        PP2["pre_process_point_history()\n→ lịch sử 16 điểm + normalize"]
        PH["point_history\ndeque(maxlen=16)"]
    end

    subgraph ClassificationLayer["🧠 Tầng Phân Loại"]
        KPC["KeyPointClassifier\n(TFLite)\ncử chỉ tĩnh"]
        PHC["PointHistoryClassifier\n(TFLite)\ncử chỉ động"]
        LABELS1["Labels:\nOpen / Close / Pointer"]
        LABELS2["Labels:\nStop / CW / CCW / Move"]
    end

    subgraph OutputLayer["🖥️ Tầng Kết Xuất"]
        DRAW[Vẽ Landmark + BRect]
        INFO[Hiển thị Text + FPS]
        IMSHOW[cv.imshow]
    end

    CAM --> CV --> MP --> LM
    MP --> BRECT
    LM --> PP1
    LM --> PH --> PP2
    PP1 --> KPC --> LABELS1
    PP2 --> PHC --> LABELS2
    BRECT --> DRAW
    LM --> DRAW
    LABELS1 --> INFO
    LABELS2 --> INFO
    DRAW --> IMSHOW
    INFO --> IMSHOW
```

---

## 2. Luồng Khởi Động Ứng Dụng

```mermaid
flowchart TD
    START([▶ python app.py]) --> ARGS

    ARGS["get_args()\nParse tham số CLI:\n--device, --width, --height\n--min_detection_confidence=0.7\n--min_tracking_confidence=0.5"]

    ARGS --> CAM_INIT

    CAM_INIT["cv.VideoCapture(device)\ncap.set WIDTH = 960\ncap.set HEIGHT = 540"]

    CAM_INIT --> MP_INIT

    MP_INIT["mp.solutions.hands.Hands(\n  static_image_mode=False,\n  max_num_hands=1,\n  min_detection_confidence=0.7,\n  min_tracking_confidence=0.5\n)"]

    MP_INIT --> MODEL_INIT

    MODEL_INIT["Khởi tạo 2 mô hình TFLite:\n① KeyPointClassifier()\n② PointHistoryClassifier()"]

    MODEL_INIT --> LABEL_LOAD

    LABEL_LOAD["Đọc CSV Labels:\n① keypoint_classifier_label.csv\n   → ['Open','Close','Pointer']\n② point_history_classifier_label.csv\n   → ['Stop','Clockwise','Counter Clockwise','Move']"]

    LABEL_LOAD --> STATE_INIT

    STATE_INIT["Khởi tạo state:\n• CvFpsCalc(buffer_len=10)\n• point_history = deque(maxlen=16)\n• finger_gesture_history = deque(maxlen=16)\n• mode = 0"]

    STATE_INIT --> MAIN_LOOP([🔄 Vào Vòng Lặp Chính])
```

---

## 3. Vòng Lặp Xử Lý Chính

```mermaid
flowchart TD
    LOOP_START([🔄 Bắt đầu vòng lặp])

    LOOP_START --> FPS["fps = cvFpsCalc.get()"]
    FPS --> KEY["key = cv.waitKey(10)"]
    KEY --> ESC{key == ESC\n27?}
    ESC -- Yes --> CLEANUP[cap.release\ncv.destroyAllWindows]
    CLEANUP --> END([⏹ Kết thúc])

    ESC -- No --> SEL_MODE["select_mode(key, mode)\n→ (number, mode)"]
    SEL_MODE --> CAP["ret, image = cap.read()"]
    CAP --> RET{ret == True?}
    RET -- No --> CLEANUP
    RET -- Yes --> FLIP["cv.flip(image, 1)\n(mirror)"]
    FLIP --> DEEP["debug_image = copy.deepcopy(image)"]
    DEEP --> BGR2RGB["cv.cvtColor BGR→RGB"]
    BGR2RGB --> HANDS["hands.process(image)\n(MediaPipe inference)"]
    HANDS --> DETECT{hand_landmarks\ndetected?}

    DETECT -- Yes --> HAND_PROC["Xử lý từng bàn tay\n(xem luồng 4)"]
    DETECT -- No --> NO_HAND["point_history.append([0, 0])"]

    HAND_PROC --> DRAW_PH["draw_point_history(debug_image,\npoint_history)"]
    NO_HAND --> DRAW_PH

    DRAW_PH --> DRAW_INFO["draw_info(debug_image, fps, mode, number)"]
    DRAW_INFO --> IMSHOW["cv.imshow('Hand Gesture Recognition',\ndebug_image)"]
    IMSHOW --> LOOP_START
```

---

## 4. Luồng Phát Hiện Tay & Trích Xuất Landmark

```mermaid
flowchart TD
    INPUT["results.multi_hand_landmarks\n(danh sách bàn tay phát hiện)"]

    INPUT --> FOR_LOOP["for hand_landmarks, handedness\nin zip(...)"]

    FOR_LOOP --> BRECT["calc_bounding_rect(\n  debug_image, hand_landmarks\n)\n→ [x1, y1, x2, y2]"]

    FOR_LOOP --> LM_CALC["calc_landmark_list(\n  debug_image, hand_landmarks\n)\n→ 21 điểm [[x,y], ...]"]

    BRECT --> PRE_PROC
    LM_CALC --> PRE_PROC

    PRE_PROC["Tiền xử lý\n(xem luồng 5)"]

    PRE_PROC --> LOG["logging_csv(\n  number, mode,\n  landmark_list,\n  point_history_list\n)"]

    LOG --> KP_CLASSIFY["keypoint_classifier(\npre_processed_landmark_list\n)\n→ hand_sign_id"]

    KP_CLASSIFY --> IS_POINTER{hand_sign_id\n== 2 (Pointer)?}

    IS_POINTER -- Yes --> APPEND_PH["point_history.append(\n  landmark_list[8]\n)\n(đầu ngón trỏ)"]

    IS_POINTER -- No --> APPEND_ZERO["point_history.append([0, 0])"]

    APPEND_PH --> PH_CLASSIFY
    APPEND_ZERO --> PH_CLASSIFY

    PH_CLASSIFY["Phân loại cử chỉ động\n(xem luồng 7)"]

    PH_CLASSIFY --> FG_HIST["finger_gesture_history.append(\n  finger_gesture_id\n)"]

    FG_HIST --> MOST_COMMON["Counter(finger_gesture_history)\n.most_common()\n→ most_common_fg_id"]

    MOST_COMMON --> DRAWING["Vẽ kết quả\n(xem luồng 9)"]
```

---

## 5. Luồng Tiền Xử Lý Dữ Liệu

### 5a. Tiền xử lý Landmark (cho KeyPoint Classifier)

```mermaid
flowchart LR
    RAW["landmark_list\n21 điểm tuyệt đối\n[[x0,y0],[x1,y1],...,[x20,y20]]"]

    RAW --> COPY["copy.deepcopy(landmark_list)"]

    COPY --> REL["Chuyển sang tọa độ tương đối:\nbase = điểm wrist [x0,y0]\nxi = xi - base_x\nyi = yi - base_y"]

    REL --> FLAT["Flatten 2D → 1D:\n[x0,y0,x1,y1,...,x20,y20]\n= 42 phần tử"]

    FLAT --> NORM["Normalize:\nmax_val = max(abs(all))\nxi = xi / max_val\n→ giá trị trong [-1.0, 1.0]"]

    NORM --> OUT1["pre_processed_landmark_list\n42 float trong [-1, 1]\n→ input KeyPointClassifier"]
```

### 5b. Tiền xử lý Point History (cho Point History Classifier)

```mermaid
flowchart LR
    RAW2["point_history\ndeque maxlen=16\n16 điểm [x,y] tuyệt đối"]

    RAW2 --> COPY2["copy.deepcopy(point_history)"]

    COPY2 --> REL2["Chuyển tọa độ tương đối:\nbase = điểm đầu tiên\nxi = (xi - base_x) / image_width\nyi = (yi - base_y) / image_height"]

    REL2 --> FLAT2["Flatten 2D → 1D:\n[x0,y0,...,x15,y15]\n= 32 phần tử"]

    FLAT2 --> OUT2["pre_processed_point_history_list\n32 float (normalized)\n→ input PointHistoryClassifier"]
```

---

## 6. Luồng Phân Loại Cử Chỉ Tĩnh (KeyPoint Classifier)

```mermaid
flowchart TD
    INIT["__init__():\nLoad keypoint_classifier.tflite\nallocate_tensors()\nlưu input_details, output_details"]

    CALL["__call__(landmark_list)\n[42 floats]"]

    INIT -.->|khởi tạo một lần| CALL

    CALL --> SET_TENSOR["interpreter.set_tensor(\n  input_index,\n  np.array([landmark_list], float32)\n)"]

    SET_TENSOR --> INVOKE["interpreter.invoke()\n(chạy inference TFLite)"]

    INVOKE --> GET_TENSOR["result = interpreter.get_tensor(\n  output_index\n)\nshape: (1, num_classes)"]

    GET_TENSOR --> ARGMAX["result_index = np.argmax(\n  np.squeeze(result)\n)"]

    ARGMAX --> RETURN["return result_index\n0=Open / 1=Close / 2=Pointer"]

    RETURN --> LABEL["keypoint_classifier_labels[result_index]\n→ tên cử chỉ hiển thị"]
```

---

## 7. Luồng Phân Loại Cử Chỉ Động (Point History Classifier)

```mermaid
flowchart TD
    INIT2["__init__():\nLoad point_history_classifier.tflite\nallocate_tensors()\nscore_th = 0.5\ninvalid_value = 0"]

    CHECK_LEN{len(pre_processed_\npoint_history_list)\n== history_length * 2\n== 32?}

    CHECK_LEN -- No --> SKIP["finger_gesture_id = 0\n(chưa đủ dữ liệu)"]

    CHECK_LEN -- Yes --> CALL2["__call__(point_history)\n[32 floats]"]

    INIT2 -.->|khởi tạo một lần| CALL2

    CALL2 --> SET_T2["interpreter.set_tensor(\n  input_index,\n  np.array([point_history], float32)\n)"]

    SET_T2 --> INV2["interpreter.invoke()"]

    INV2 --> GET_T2["result = interpreter.get_tensor(\n  output_index\n)"]

    GET_T2 --> ARG2["result_index = np.argmax(\n  np.squeeze(result)\n)"]

    ARG2 --> THRESH{score[result_index]\n>= score_th\n(0.5)?}

    THRESH -- No --> INVALID["result_index = invalid_value (0)\n→ 'Stop'"]

    THRESH -- Yes --> VALID["return result_index\n0=Stop / 1=CW / 2=CCW / 3=Move"]

    SKIP --> FG_OUT
    INVALID --> FG_OUT
    VALID --> FG_OUT

    FG_OUT["finger_gesture_id\n→ finger_gesture_history.append()\n→ most_common() voting"]
```

---

## 8. Luồng Quản Lý Chế Độ & Ghi Dữ Liệu Huấn Luyện

### 8a. Quản lý phím bấm

```mermaid
flowchart TD
    KEY_INPUT["key = cv.waitKey(10)"]

    KEY_INPUT --> DIGIT{48 ≤ key ≤ 57\n(phím '0'-'9')?}
    DIGIT -- Yes --> NUM["number = key - 48\n(0 đến 9)"]
    DIGIT -- No --> NUM0["number = -1"]

    NUM --> MODE_CHECK
    NUM0 --> MODE_CHECK

    MODE_CHECK --> KEY_N{key == 110\n'n'?}
    KEY_N -- Yes --> MODE0["mode = 0\n(Normal mode)"]
    KEY_N -- No --> KEY_K{key == 107\n'k'?}

    KEY_K -- Yes --> MODE1["mode = 1\n(Logging Keypoint)"]
    KEY_K -- No --> KEY_H{key == 104\n'h'?}

    KEY_H -- Yes --> MODE2["mode = 2\n(Logging Point History)"]
    KEY_H -- No --> UNCHANGED["mode không đổi"]

    MODE0 --> RETURN_MODE["return number, mode"]
    MODE1 --> RETURN_MODE
    MODE2 --> RETURN_MODE
    UNCHANGED --> RETURN_MODE
```

### 8b. Ghi dữ liệu CSV

```mermaid
flowchart TD
    LOG_CALL["logging_csv(\n  number, mode,\n  landmark_list,\n  point_history_list\n)"]

    LOG_CALL --> MODE_CHK{mode?}

    MODE_CHK -- "0 (Normal)" --> PASS["pass\n(không ghi)"]

    MODE_CHK -- "1 (Keypoint)" --> CHK1{0 ≤ number ≤ 9?}
    CHK1 -- Yes --> WRITE1["Ghi vào:\nkeypoint_classifier/keypoint.csv\nformat: [number, x0,y0,...,x20,y20]"]
    CHK1 -- No --> PASS

    MODE_CHK -- "2 (History)" --> CHK2{0 ≤ number ≤ 9?}
    CHK2 -- Yes --> WRITE2["Ghi vào:\npoint_history_classifier/point_history.csv\nformat: [number, x0,y0,...,x15,y15]"]
    CHK2 -- No --> PASS
```

---

## 9. Luồng Kết Xuất Hình Ảnh

```mermaid
flowchart TD
    INPUTS(["Đầu vào:\ndebug_image, brect,\nhandedness, hand_sign_id,\nmost_common_fg_id"])

    INPUTS --> D1["draw_bounding_rect(\n  use_brect=True,\n  debug_image, brect\n)\n→ vẽ cv.rectangle đen"]

    D1 --> D2["draw_landmarks(\n  debug_image, landmark_list\n)\n• Vẽ 16 đường kết nối ngón tay\n  (màu trắng, viền đen)\n• Vẽ 21 vòng tròn keypoint\n  (fingertip: r=8, khác: r=5)"]

    D2 --> D3["draw_info_text(\n  debug_image, brect,\n  handedness,\n  keypoint_classifier_labels[hand_sign_id],\n  point_history_classifier_labels[fg_id]\n)\n→ 'Left/Right:Open'\n→ 'Finger Gesture:Clockwise'"]

    D3 --> D4["draw_point_history(\n  debug_image, point_history\n)\n→ vẽ trail ngón trỏ\n  (màu xanh lá, bán kính tăng dần)"]

    D4 --> D5["draw_info(\n  debug_image, fps, mode, number\n)\n→ 'FPS: 30.5' (góc trái trên)\n→ 'MODE: Logging Key Point' (mode 1/2)\n→ 'NUM: 3' (nếu có phím số)"]

    D5 --> SHOW["cv.imshow(\n  'Hand Gesture Recognition',\n  debug_image\n)"]
```

---

## 10. Luồng Xử Lý Toàn Bộ Một Frame

Sơ đồ tổng hợp toàn bộ pipeline xử lý từ một frame camera đến hình ảnh kết quả:

```mermaid
sequenceDiagram
    participant CAM as 📷 Camera
    participant OCV as OpenCV
    participant MP as MediaPipe Hands
    participant PRE as Pre-processor
    participant KPC as KeyPoint Classifier
    participant PHC as PointHistory Classifier
    participant HIST as gesture_history
    participant DRAW as Renderer

    CAM->>OCV: frame BGR (960×540)
    OCV->>OCV: flip(horizontal mirror)
    OCV->>OCV: cvtColor BGR→RGB
    OCV->>MP: process(image)
    MP-->>OCV: multi_hand_landmarks, handedness

    alt Phát hiện được bàn tay
        OCV->>PRE: hand_landmarks
        PRE->>PRE: calc_bounding_rect() → [x1,y1,x2,y2]
        PRE->>PRE: calc_landmark_list() → 21 [x,y]
        PRE->>PRE: pre_process_landmark() → 42 floats [-1,1]
        PRE->>PRE: pre_process_point_history() → 32 floats

        PRE->>KPC: 42 floats
        KPC->>KPC: TFLite inference
        KPC-->>PRE: hand_sign_id (0/1/2)

        alt hand_sign_id == 2 (Pointer)
            PRE->>HIST: append(landmark[8]) ← vị trí đầu ngón trỏ
        else
            PRE->>HIST: append([0, 0])
        end

        alt len(point_history) == 32
            PRE->>PHC: 32 floats
            PHC->>PHC: TFLite inference + threshold (0.5)
            PHC-->>HIST: finger_gesture_id
        end

        HIST->>HIST: Counter.most_common()
        HIST-->>DRAW: most_common_fg_id

        PRE-->>DRAW: brect, landmark_list, handedness
        KPC-->>DRAW: hand_sign_id
    else Không phát hiện
        OCV->>HIST: append([0,0])
    end

    DRAW->>DRAW: draw_bounding_rect()
    DRAW->>DRAW: draw_landmarks()
    DRAW->>DRAW: draw_info_text()
    DRAW->>DRAW: draw_point_history()
    DRAW->>DRAW: draw_info(fps, mode)
    DRAW-->>OCV: debug_image kết quả
    OCV->>OCV: cv.imshow()
```

---

## 11. Sơ Đồ Thành Phần Hệ Thống

```mermaid
graph LR
    subgraph app["app.py — Điều phối chính"]
        MAIN[main()]
        SEL[select_mode()]
        CALC_BR[calc_bounding_rect()]
        CALC_LM[calc_landmark_list()]
        PP_LM[pre_process_landmark()]
        PP_PH[pre_process_point_history()]
        LOG_CSV[logging_csv()]
        subgraph draw_funcs["Hàm vẽ"]
            DR_BR[draw_bounding_rect()]
            DR_LM[draw_landmarks()]
            DR_IT[draw_info_text()]
            DR_PH[draw_point_history()]
            DR_IN[draw_info()]
        end
    end

    subgraph models["model/"]
        subgraph kp["keypoint_classifier/"]
            KPC_PY[keypoint_classifier.py]
            KPC_TF[keypoint_classifier.tflite]
            KPC_CSV[keypoint_classifier_label.csv]
            KP_DATA[keypoint.csv — training data]
        end
        subgraph ph["point_history_classifier/"]
            PHC_PY[point_history_classifier.py]
            PHC_TF[point_history_classifier.tflite]
            PHC_CSV[point_history_classifier_label.csv]
            PH_DATA[point_history.csv — training data]
        end
    end

    subgraph utils["utils/"]
        FPS[cvfpscalc.py\nCvFpsCalc]
    end

    subgraph ext["Thư viện ngoài"]
        CV2[OpenCV cv2]
        MEDIAPIPE[MediaPipe]
        TFLITE[TFLite Runtime]
        NUMPY[NumPy]
    end

    MAIN --> SEL
    MAIN --> CALC_BR
    MAIN --> CALC_LM
    MAIN --> PP_LM
    MAIN --> PP_PH
    MAIN --> LOG_CSV
    MAIN --> draw_funcs
    MAIN --> FPS

    KPC_PY --> KPC_TF
    KPC_PY --> TFLITE
    PHC_PY --> PHC_TF
    PHC_PY --> TFLITE

    MAIN --> KPC_PY
    MAIN --> PHC_PY
    MAIN --> CV2
    MAIN --> MEDIAPIPE
    MAIN --> NUMPY

    LOG_CSV -->|mode=1| KP_DATA
    LOG_CSV -->|mode=2| PH_DATA
    KPC_PY -->|đọc labels| KPC_CSV
    PHC_PY -->|đọc labels| PHC_CSV
```

---

## 12. Sơ Đồ Trạng Thái Chế Độ

```mermaid
stateDiagram-v2
    [*] --> Normal : Khởi động\nmode = 0

    Normal : Mode 0 — Normal\nChỉ nhận diện cử chỉ\nKhông ghi dữ liệu

    LoggingKeypoint : Mode 1 — Logging Keypoint\nNhấn phím 0-9 để gán nhãn\nGhi 42 floats → keypoint.csv

    LoggingHistory : Mode 2 — Logging History\nNhấn phím 0-9 để gán nhãn\nGhi 32 floats → point_history.csv

    Normal --> LoggingKeypoint : phím 'k' (107)
    Normal --> LoggingHistory : phím 'h' (104)
    LoggingKeypoint --> Normal : phím 'n' (110)
    LoggingKeypoint --> LoggingHistory : phím 'h' (104)
    LoggingHistory --> Normal : phím 'n' (110)
    LoggingHistory --> LoggingKeypoint : phím 'k' (107)

    Normal --> [*] : ESC (27)
    LoggingKeypoint --> [*] : ESC (27)
    LoggingHistory --> [*] : ESC (27)
```

---

## 13. Luồng Pipeline Mô Hình TFLite

```mermaid
flowchart LR
    subgraph training["🏋️ Pha Huấn Luyện (offline)"]
        RAW_DATA["Dữ liệu thô\n(thu thập qua mode 1/2)"]
        NB1["keypoint_classification.ipynb\n→ train Keras model\n→ export .tflite + .hdf5"]
        NB2["point_history_classification.ipynb\n→ train Keras model\n→ export .tflite + .hdf5"]
        RAW_DATA --> NB1
        RAW_DATA --> NB2
    end

    subgraph models_files["📁 File mô hình"]
        TF1["keypoint_classifier.tflite\nInput: (1, 42)\nOutput: (1, 3) — 3 lớp"]
        TF2["point_history_classifier.tflite\nInput: (1, 32)\nOutput: (1, 4) — 4 lớp"]
        NB1 --> TF1
        NB2 --> TF2
    end

    subgraph inference["⚡ Pha Suy Luận (real-time)"]
        IN1["42 floats\n(21 landmark × 2)"]
        IN2["32 floats\n(16 điểm lịch sử × 2)"]
        TF1 --> OUT1["Softmax scores\nnp.argmax → class_id"]
        TF2 --> OUT2["Softmax scores\n+ threshold 0.5\nnp.argmax → class_id"]
        IN1 --> TF1
        IN2 --> TF2
    end

    subgraph output["🏷️ Kết quả"]
        L1["Open / Close / Pointer"]
        L2["Stop / Clockwise\nCounter CW / Move"]
        OUT1 --> L1
        OUT2 --> L2
    end
```

---

## Tóm Tắt Luồng Dữ Liệu Tổng Quát

| Bước | Đầu vào         | Xử lý                             | Đầu ra                          |
| ---- | --------------- | --------------------------------- | ------------------------------- |
| 1    | Camera frame    | `cv.VideoCapture.read()` + flip   | BGR image 960×540               |
| 2    | BGR image       | `cvtColor BGR→RGB`                | RGB image                       |
| 3    | RGB image       | `MediaPipe Hands.process()`       | 21 landmark + handedness        |
| 4    | 21 landmark     | `calc_landmark_list()`            | 21 điểm `[x,y]` tuyệt đối       |
| 5    | 21 điểm pixel   | `pre_process_landmark()`          | 42 floats trong `[-1, 1]`       |
| 6    | 42 floats       | `KeyPointClassifier` (TFLite)     | `hand_sign_id` ∈ {0,1,2}        |
| 7    | 16 điểm lịch sử | `pre_process_point_history()`     | 32 floats normalized            |
| 8    | 32 floats       | `PointHistoryClassifier` (TFLite) | `finger_gesture_id` ∈ {0,1,2,3} |
| 9    | gesture_id × 16 | `Counter.most_common()`           | gesture ổn định nhất            |
| 10   | Tất cả kết quả  | `draw_*()` functions              | Frame kết xuất + text           |

---

_Báo cáo được tạo tự động từ phân tích mã nguồn dự án hand-gesture-recognition-using-mediapipe._
