#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
import os
from collections import Counter
from collections import deque

# Force X11/xcb backend so Qt keyboard events work under Wayland
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from utils.class_menu import ClassMenu
from utils.collection_manager import CollectionManager, HandData
from utils.feature_extractor import FeatureExtractor
from model import KeyPointClassifierV2
from model import PointHistoryClassifier


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help="cap width", type=int, default=1280)
    parser.add_argument("--height", help="cap height", type=int, default=720)

    parser.add_argument("--use_static_image_mode", action="store_true")
    parser.add_argument(
        "--min_detection_confidence",
        help="min_detection_confidence",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--min_tracking_confidence",
        help="min_tracking_confidence",
        type=int,
        default=0.5,
    )

    args = parser.parse_args()

    return args


def main():
    # 引数解析 #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    # Verify actual resolution
    actual_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    if actual_w != cap_width or actual_h != cap_height:
        print(f"[WARN] Requested {cap_width}x{cap_height}, got {actual_w}x{actual_h}")

    # モデルロード #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifierV2()
    feature_extractor = FeatureExtractor()

    point_history_classifier = PointHistoryClassifier()

    # ラベル読み込み ###########################################################
    with open(
        "model/keypoint_classifier/keypoint_classifier_v2_label.csv",
        encoding="utf-8-sig",
    ) as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]
    with open(
        "model/point_history_classifier/point_history_classifier_label.csv",
        encoding="utf-8-sig",
    ) as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # 座標履歴 #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # フィンガージェスチャー履歴 ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    # Collection UI components
    label_csv_path = "model/keypoint_classifier/keypoint_classifier_v2_label.csv"
    class_menu = ClassMenu(label_csv_path)
    collection_mgr = CollectionManager(
        csv_path="model/keypoint_classifier/keypoint.csv",
    )
    class_menu.set_class_counts(collection_mgr.class_counts)

    # Point history logging (legacy mode, activated by 'h')
    point_history_mode = False
    point_history_class = -1

    while True:
        fps = cvFpsCalc.get()

        # キー処理(ESC：終了) #################################################
        key = cv.waitKeyEx(10)

        # Collection manager tick (auto-transitions: done → idle)
        collection_mgr.tick()
        class_menu.set_class_counts(collection_mgr.class_counts)

        if key == 27:  # ESC
            if collection_mgr.state in ("countdown", "recording"):
                collection_mgr.cancel()
            elif class_menu.visible:
                class_menu.toggle()  # close menu
            else:
                break  # quit app
        elif key == 9:  # Tab
            if collection_mgr.state == "idle":
                class_menu.toggle()
        elif key == 65362:  # Up arrow
            class_menu.move_up()
        elif key == 65364:  # Down arrow
            class_menu.move_down()
        elif key == 13:  # Enter
            class_id = class_menu.confirm()
            if class_id is not None:
                collection_mgr.start_session(class_id)
        elif key == 32:  # Space
            if collection_mgr.state in ("countdown", "recording"):
                collection_mgr.cancel()
        elif key == ord("+") or key == ord("="):
            if class_menu.visible:
                collection_mgr.adjust_batch_size(10)
        elif key == ord("-"):
            if class_menu.visible:
                collection_mgr.adjust_batch_size(-10)
        # Legacy point-history logging
        elif key == 104:  # h
            point_history_mode = not point_history_mode
        elif point_history_mode and 48 <= key <= 57:
            point_history_class = key - 48

        # カメラキャプチャ #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # ミラー表示
        debug_image = copy.deepcopy(image)

        # 検出実施 #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        hands_data = []  # for CollectionManager
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                # 外接矩形の計算
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # ランドマークの計算
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # 相対座標・正規化座標への変換
                pre_processed_landmark_list = feature_extractor.extract(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history
                )

                # Collect hand data for guided collection
                confidence = handedness.classification[0].score
                hands_data.append(
                    HandData(
                        features=pre_processed_landmark_list, confidence=confidence
                    )
                )

                # Legacy point-history logging
                if point_history_mode and 0 <= point_history_class <= 9:
                    logging_csv_point_history(
                        point_history_class, pre_processed_point_history_list
                    )

                # ハンドサイン分類
                hand_sign_id, _ = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # 指差しサイン
                    point_history.append(landmark_list[8][0:2])  # 人差指座標
                else:
                    point_history.append([0, 0])

                # フィンガージェスチャー分類
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list
                    )

                # 直近検出の中で最多のジェスチャーIDを算出
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()

                # 描画
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(
                    debug_image, [p[:2] for p in landmark_list]
                )
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            point_history.append([0, 0])

        # Feed hands to collection manager
        if collection_mgr.state in ("countdown", "recording"):
            collection_mgr.on_frame(hands_data)

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_collection_overlay(
            debug_image,
            collection_mgr,
            keypoint_classifier_labels,
        )
        debug_image = draw_balance_chart(
            debug_image,
            collection_mgr.class_counts,
            keypoint_classifier_labels,
        )
        debug_image = draw_info(debug_image, fps, collection_mgr)
        debug_image = class_menu.draw(debug_image)

        # 画面反映 #############################################################
        cv.imshow("Hand Gesture Recognition", debug_image)

    cap.release()
    cv.destroyAllWindows()


def logging_csv_point_history(number, point_history_list):
    """Legacy: append point-history row to CSV."""
    if 0 <= number <= 9:
        csv_path = "model/point_history_classifier/point_history.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # キーポイント
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y, landmark_z])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # 相対座標に変換
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # 1次元リストに変換
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # 正規化
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # 相対座標に変換
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (
            temp_point_history[index][0] - base_x
        ) / image_width
        temp_point_history[index][1] = (
            temp_point_history[index][1] - base_y
        ) / image_height

    # 1次元リストに変換
    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def draw_balance_chart(image, class_counts, labels):
    """Draw mini horizontal bar chart of per-class sample counts at top-right."""
    if not class_counts:
        return image
    h, w = image.shape[:2]
    max_count = max(class_counts.values()) if class_counts else 1
    bar_max_w = 120
    line_h = 18
    pad = 8
    num_classes = len(labels)
    chart_h = line_h * num_classes + pad * 2
    chart_w = bar_max_w + 140
    chart_x = w - chart_w - 10
    chart_y = 10

    # Semi-transparent background
    overlay = image.copy()
    cv.rectangle(
        overlay,
        (chart_x, chart_y),
        (chart_x + chart_w, chart_y + chart_h),
        (20, 20, 20),
        -1,
    )
    cv.addWeighted(overlay, 0.75, image, 0.25, 0, image)

    for i, label in enumerate(labels):
        count = class_counts.get(i, 0)
        y = chart_y + pad + i * line_h + 14
        # Label (truncated)
        short = label[:8]
        cv.putText(
            image,
            short,
            (chart_x + pad, y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.35,
            (180, 180, 180),
            1,
        )
        # Bar
        bar_x = chart_x + 75
        bar_w = int(bar_max_w * count / max(max_count, 1))
        color = (0, 180, 0) if count >= 100 else (0, 140, 255)
        cv.rectangle(image, (bar_x, y - 10), (bar_x + bar_w, y - 2), color, -1)
        # Count text + NEEDS MORE indicator
        count_label = str(count)
        if count < 100:
            count_label += " NEEDS MORE"
            count_color = (0, 140, 255)
        else:
            count_color = (160, 160, 160)
        cv.putText(
            image,
            count_label,
            (bar_x + bar_max_w + 5, y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.35,
            count_color,
            1,
        )
    return image


def draw_collection_overlay(image, mgr, labels):
    """Draw countdown / recording / done overlay based on CollectionManager state."""
    info = mgr.get_overlay_state()
    state = info["state"]
    h, w = image.shape[:2]

    if state == "countdown":
        remaining = info.get("countdown_remaining", 0)
        class_id = info["class_id"]
        label = labels[class_id] if class_id < len(labels) else str(class_id)
        # Large centered countdown number
        count_text = str(int(remaining) + 1)
        text_size = cv.getTextSize(count_text, cv.FONT_HERSHEY_SIMPLEX, 4.0, 6)[0]
        tx = (w - text_size[0]) // 2
        ty = (h + text_size[1]) // 2
        # Semi-transparent backdrop
        overlay = image.copy()
        cv.rectangle(
            overlay,
            (tx - 30, ty - text_size[1] - 30),
            (tx + text_size[0] + 30, ty + 50),
            (0, 0, 0),
            -1,
        )
        cv.addWeighted(overlay, 0.5, image, 0.5, 0, image)
        cv.putText(
            image,
            count_text,
            (tx, ty),
            cv.FONT_HERSHEY_SIMPLEX,
            4.0,
            (0, 255, 255),
            6,
            cv.LINE_AA,
        )
        # Class label below
        sub = f"Class {class_id}: {label}"
        sub_size = cv.getTextSize(sub, cv.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv.putText(
            image,
            sub,
            ((w - sub_size[0]) // 2, ty + 40),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (200, 200, 200),
            2,
            cv.LINE_AA,
        )
        # Pose instruction
        hint = "Giu tay dung pose!"
        hint_size = cv.getTextSize(hint, cv.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv.putText(
            image,
            hint,
            ((w - hint_size[0]) // 2, ty + 70),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            1,
            cv.LINE_AA,
        )

    elif state == "recording":
        class_id = info["class_id"]
        collected = info["collected"]
        target = info["target"]
        label = labels[class_id] if class_id < len(labels) else str(class_id)
        # Green border
        cv.rectangle(image, (0, 0), (w - 1, h - 1), (0, 220, 0), 4)
        # Top banner
        banner_h = 50
        overlay = image.copy()
        cv.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
        cv.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        rec_text = f"[REC \u25cf] {label}  {collected}/{target}"
        cv.putText(
            image,
            rec_text,
            (15, 35),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 220, 0),
            2,
            cv.LINE_AA,
        )
        # Progress bar
        bar_x, bar_y, bar_w, bar_h = 15, 42, w - 30, 6
        progress = min(collected / max(target, 1), 1.0)
        cv.rectangle(
            image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1
        )
        cv.rectangle(
            image,
            (bar_x, bar_y),
            (bar_x + int(bar_w * progress), bar_y + bar_h),
            (0, 220, 0),
            -1,
        )

    elif state == "done":
        flushed = info.get("flushed_count", 0)
        target = info.get("flushed_target", 0)
        timed_out = info.get("timed_out", False)
        done_class_id = info.get("class_id", -1)
        done_label = labels[done_class_id] if 0 <= done_class_id < len(labels) else ""
        if timed_out:
            msg = f"\u26a0 Timeout: {flushed}/{target} frames saved for {done_label}"
            color = (0, 180, 255)
        else:
            msg = f"\u2713 {flushed} frames saved for {done_label}"
            color = (0, 220, 0)
        text_size = cv.getTextSize(msg, cv.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        tx = (w - text_size[0]) // 2
        ty = (h + text_size[1]) // 2
        cv.putText(
            image, msg, (tx, ty), cv.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv.LINE_AA
        )

    return image


def draw_landmarks(image, landmark_point):
    # 接続線
    if len(landmark_point) > 0:
        # 親指
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[2]),
            tuple(landmark_point[3]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[3]),
            tuple(landmark_point[4]),
            (255, 255, 255),
            2,
        )

        # 人差指
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[5]),
            tuple(landmark_point[6]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[6]),
            tuple(landmark_point[7]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[7]),
            tuple(landmark_point[8]),
            (255, 255, 255),
            2,
        )

        # 中指
        cv.line(
            image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[9]),
            tuple(landmark_point[10]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[10]),
            tuple(landmark_point[11]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[11]),
            tuple(landmark_point[12]),
            (255, 255, 255),
            2,
        )

        # 薬指
        cv.line(
            image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[13]),
            tuple(landmark_point[14]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[14]),
            tuple(landmark_point[15]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[15]),
            tuple(landmark_point[16]),
            (255, 255, 255),
            2,
        )

        # 小指
        cv.line(
            image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[17]),
            tuple(landmark_point[18]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[18]),
            tuple(landmark_point[19]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[19]),
            tuple(landmark_point[20]),
            (255, 255, 255),
            2,
        )

        # 手の平
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[0]),
            tuple(landmark_point[1]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[1]),
            tuple(landmark_point[2]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[2]),
            tuple(landmark_point[5]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[5]),
            tuple(landmark_point[9]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[9]),
            tuple(landmark_point[13]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[13]),
            tuple(landmark_point[17]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[17]),
            tuple(landmark_point[0]),
            (255, 255, 255),
            2,
        )

    # キーポイント
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # 外接矩形
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ":" + hand_sign_text
    cv.putText(
        image,
        info_text,
        (brect[0] + 5, brect[1] - 4),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )

    if finger_gesture_text != "":
        cv.putText(
            image,
            "Finger Gesture:" + finger_gesture_text,
            (10, 60),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            4,
            cv.LINE_AA,
        )
        cv.putText(
            image,
            "Finger Gesture:" + finger_gesture_text,
            (10, 60),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(
                image, (point[0], point[1]), 1 + int(index / 2), (152, 251, 152), 2
            )

    return image


def draw_info(image, fps, collection_mgr):
    cv.putText(
        image,
        "FPS:" + str(fps),
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        4,
        cv.LINE_AA,
    )
    cv.putText(
        image,
        "FPS:" + str(fps),
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )

    if collection_mgr.state == "idle":
        hint = "Tab=class menu  +/-=batch size"
        h = image.shape[0]
        cv.putText(
            image,
            hint,
            (10, h - 20),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (160, 160, 160),
            1,
            cv.LINE_AA,
        )
        batch_text = f"Batch: {collection_mgr.batch_size}"
        cv.putText(
            image,
            batch_text,
            (10, h - 45),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (160, 160, 160),
            1,
            cv.LINE_AA,
        )
    return image


if __name__ == "__main__":
    main()
