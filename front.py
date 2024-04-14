# import cv2 as cv
# import numpy as np
# import streamlit as st
# import mediapipe as mp
# from model import KeyPointClassifier, PointHistoryClassifier
# from collections import Counter, deque
# from utils import CvFpsCalc
# import csv
# import itertools
# import copy


# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=2,
#     min_detection_confidence=0.7,
#     min_tracking_confidence=0.5,
# )

# keypoint_classifier = KeyPointClassifier()
# point_history_classifier = PointHistoryClassifier()

# # Load model labels
# with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
#     keypoint_classifier_labels = csv.reader(f)
#     keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

# with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
#     point_history_classifier_labels = csv.reader(f)
#     point_history_classifier_labels = [row[0] for row in point_history_classifier_labels]

# cvFpsCalc = CvFpsCalc(buffer_len=10)
# history_length = 16
# point_history = deque(maxlen=history_length)
# finger_gesture_history = deque(maxlen=history_length)
# predictions = ""

# st.title("Hand Gesture Recognition")

# predType = st.radio("Choose prediction type", ("Webcam", "Pre-recorded video"))
# if predType == "Webcam":
#     cap = cv.VideoCapture(0)
# else:
#     video_file = st.file_uploader("Upload video file", type=["mp4", "avi"])
#     if video_file is not None:
#         tfile = tempfile.NamedTemporaryFile(delete=False)
#         tfile.write(video_file.read())

#         cap = cv.VideoCapture(tfile.name)

# while cap.isOpened():
#     ret, frame = cap.read()

#     if not ret:
#         st.warning("Unable to read frame from video source.")
#         break

#     frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
#     frame.flags.writeable = False
#     results = hands.process(frame)
#     frame.flags.writeable = True

#     if results.multi_hand_landmarks is not None:
#         left_hand_landmarks = None
#         right_hand_landmarks = None

#         for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
#             brect = calc_bounding_rect(frame, hand_landmarks)
#             landmark_list = calc_landmark_list(frame, hand_landmarks)

#             if handedness.classification[0].label == "Left":
#                 left_hand_landmarks = landmark_list
#             elif handedness.classification[0].label == "Right":
#                 right_hand_landmarks = landmark_list

#             left_hand_landmarks = left_hand_landmarks if left_hand_landmarks is not None else [[0, 0]] * 21
#             right_hand_landmarks = right_hand_landmarks if right_hand_landmarks is not None else [[0, 0]] * 21

#             landmark_list = left_hand_landmarks + right_hand_landmarks

#             prl1 = pre_process_landmark(landmark_list[:21])
#             prl2 = pre_process_landmark(landmark_list[21:])
#             pre_processed_landmark_list = prl1 + prl2

#             pre_processed_point_history_list = pre_process_point_history(frame, point_history)

#             hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
#             if hand_sign_id == 2:  # Pointing sign
#                 point_history.append(landmark_list[8])  # Index finger coordinates
#             else:
#                 point_history.append([0, 0])

#             finger_gesture_id = 0
#             point_history_len = len(pre_processed_point_history_list)
#             if point_history_len == (history_length * 2):
#                 finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

#             finger_gesture_history.append(finger_gesture_id)
#             most_common_fg_id = Counter(finger_gesture_history).most_common()

#             frame = draw_bounding_rect(True, frame, brect)
#             frame = draw_landmarks(frame, landmark_list)
#             frame = draw_info_text(frame, brect, handedness,
#                                    keypoint_classifier_labels[hand_sign_id],
#                                    point_history_classifier_labels[most_common_fg_id[0][0]])

#             if hand_sign_id != -1 and keypoint_classifier_labels[hand_sign_id] != predictions:
#                 predictions += keypoint_classifier_labels[hand_sign_id] + " "

#     else:
#         point_history.append([0, 0])

#     frame = draw_point_history(frame, point_history)
#     frame = draw_info(frame, cvFpsCalc.get_fps(), 0, -1)

#     st.image(frame)

#     if cv.waitKey(1) == 27:  # ESC
#         break

# cap.release()
# cv.destroyAllWindows()

# if predType != "Webcam":
#     st.write("Output sentence:")
#     st.write(predictions)

# import cv2 as cv
# import numpy as np
# import streamlit as st
# import mediapipe as mp
# from model import KeyPointClassifier, PointHistoryClassifier
# from collections import Counter, deque
# from utils import CvFpsCalc
# import csv
# import itertools
# import copy

# def calc_bounding_rect(image, landmarks):
#     image_width, image_height = image.shape[1], image.shape[0]

#     landmark_array = np.empty((0, 2), int)

#     for _, landmark in enumerate(landmarks.landmark):
#         landmark_x = min(int(landmark.x * image_width), image_width - 1)
#         landmark_y = min(int(landmark.y * image_height), image_height - 1)

#         landmark_point = [np.array((landmark_x, landmark_y))]

#         landmark_array = np.append(landmark_array, landmark_point, axis=0)

#     x, y, w, h = cv.boundingRect(landmark_array)

#     return [x, y, x + w, y + h]

# def calc_landmark_list(image, landmarks):
#     image_width, image_height = image.shape[1], image.shape[0]

#     landmark_point = []

#     # キーポイント
#     for _, landmark in enumerate(landmarks.landmark):
#         landmark_x = min(int(landmark.x * image_width), image_width - 1)
#         landmark_y = min(int(landmark.y * image_height), image_height - 1)
#         # landmark_z = landmark.z

#         landmark_point.append([landmark_x, landmark_y])

#     return landmark_point

# def pre_process_landmark(landmark_list):
#     temp_landmark_list = copy.deepcopy(landmark_list)

#     # 相対座標に変換
#     base_x, base_y = 0, 0
#     for index, landmark_point in enumerate(temp_landmark_list):
#         if index == 0:
#             base_x, base_y = landmark_point[0], landmark_point[1]

#         temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
#         temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

#     # 1次元リストに変換
#     temp_landmark_list = list(
#         itertools.chain.from_iterable(temp_landmark_list))

#     # 正規化
#     max_value = max(list(map(abs, temp_landmark_list)))

#     def normalize_(n):
#         if n == 0 or max_value == 0:
#             return 0
#         return n / max_value

#     temp_landmark_list = list(map(normalize_, temp_landmark_list))

#     return temp_landmark_list


# def pre_process_point_history(image, point_history):
#     image_width, image_height = image.shape[1], image.shape[0]

#     temp_point_history = copy.deepcopy(point_history)

#     # 相対座標に変換
#     base_x, base_y = 0, 0
#     for index, point in enumerate(temp_point_history):
#         if index == 0:
#             base_x, base_y = point[0], point[1]

#         temp_point_history[index][0] = (temp_point_history[index][0] -
#                                         base_x) / image_width
#         temp_point_history[index][1] = (temp_point_history[index][1] -
#                                         base_y) / image_height

#     # 1次元リストに変換
#     temp_point_history = list(
#         itertools.chain.from_iterable(temp_point_history))

#     return temp_point_history


# def logging_csv(number, mode, landmark_list, point_history_list):
#     if mode == 0:
#         pass
#     if mode == 1:
#         # csv_path = 'model/keypoint_classifier/keypoint.csv'
#         csv_path = 'model/keypoint_classifier/alphakeypoints.csv'
#         with open(csv_path, 'a', newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow([24, *landmark_list])
#     if mode == 2 and (0 <= number <= 9):
#         csv_path = 'model/point_history_classifier/point_history.csv'
#         with open(csv_path, 'a', newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow([number, *point_history_list])
#     return


# def draw_landmarks(image, landmark_point):
#     # 接続線
#     if len(landmark_point) > 0:
#         # 親指
#         cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
#                 (255, 255, 255), 2)

#         # 人差指
#         cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
#                 (255, 255, 255), 2)

#         # 中指
#         cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
#                 (255, 255, 255), 2)

#         # 薬指
#         cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
#                 (255, 255, 255), 2)

#         # 小指
#         cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
#                 (255, 255, 255), 2)

#         # 手の平
#         cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
#                 (255, 255, 255), 2)

#     # キーポイント
#     for index, landmark in enumerate(landmark_point):
#         if index == 0:  # 手首1
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 1:  # 手首2
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 2:  # 親指：付け根
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 3:  # 親指：第1関節
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 4:  # 親指：指先
#             cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
#         if index == 5:  # 人差指：付け根
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 6:  # 人差指：第2関節
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 7:  # 人差指：第1関節
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 8:  # 人差指：指先
#             cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
#         if index == 9:  # 中指：付け根
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 10:  # 中指：第2関節
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 11:  # 中指：第1関節
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 12:  # 中指：指先
#             cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
#         if index == 13:  # 薬指：付け根
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 14:  # 薬指：第2関節
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 15:  # 薬指：第1関節
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 16:  # 薬指：指先
#             cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
#         if index == 17:  # 小指：付け根
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 18:  # 小指：第2関節
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 19:  # 小指：第1関節
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 20:  # 小指：指先
#             cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

#     return image


# def draw_bounding_rect(use_brect, image, brect):
#     if use_brect:
#         # 外接矩形
#         cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
#                      (0, 0, 0), 1)

#     return image


# def draw_info_text(image, brect, handedness, hand_sign_text,
#                    finger_gesture_text):
#     cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
#                  (0, 0, 0), -1)

#     info_text = handedness.classification[0].label[0:]
#     if hand_sign_text != "":
#         info_text = info_text + ':' + hand_sign_text
#     cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
#                cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

#     if finger_gesture_text != "":
#         cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
#                    cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
#         cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
#                    cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
#                    cv.LINE_AA)

#     return image


# def draw_point_history(image, point_history):
#     # for index, point in enumerate(point_history):
#     #     if point[0] != 0 and point[1] != 0:
#     #         cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
#     #                   (152, 251, 152), 2)

#     return image


# def draw_info(image, fps, mode, number):
#     cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
#                1.0, (0, 0, 0), 4, cv.LINE_AA)
#     cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
#                1.0, (255, 255, 255), 2, cv.LINE_AA)

#     mode_string = ['Logging Key Point', 'Logging Point History']
#     if 1 <= mode <= 2:
#         cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
#                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
#                    cv.LINE_AA)
#         if 0 <= number <= 9:
#             cv.putText(image, "NUM:" + str(number), (10, 110),
#                        cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
#                        cv.LINE_AA)
#     return image

# # Initialize CvFpsCalc object
# cvFpsCalc = CvFpsCalc(buffer_len=10)

# # Initialize MediaPipe Hands object
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=2,
#     min_detection_confidence=0.7,
#     min_tracking_confidence=0.5,
# )

# keypoint_classifier = KeyPointClassifier()
# point_history_classifier = PointHistoryClassifier()

# # Load model labels (same as before)

# history_length = 16
# point_history = deque(maxlen=history_length)
# finger_gesture_history = deque(maxlen=history_length)
# predictions = ""

# st.title("Hand Gesture Recognition")

# # Radio button to choose prediction type
# predType = st.radio("Choose prediction type", ("Webcam", "Pre-recorded video"))

# # Main logic
# if predType == "Webcam":
#     # Open webcam
#     cap = cv.VideoCapture(0)
# else:
#     # Allow user to upload video file
#     video_file = st.file_uploader("Upload video file", type=["mp4", "avi"])
#     if video_file is not None:
#         tfile = tempfile.NamedTemporaryFile(delete=False)
#         tfile.write(video_file.read())
#         cap = cv.VideoCapture(tfile.name)

# # Main loop
# while cap.isOpened():
#     ret, frame = cap.read()

#     if not ret:
#         st.warning("Unable to read frame from video source.")
#         break

#     frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
#     frame.flags.writeable = False
#     results = hands.process(frame)
#     frame.flags.writeable = True
#     if results.multi_hand_landmarks is not None:
#         left_hand_landmarks = None
#         right_hand_landmarks = None

#         for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
#             brect = calc_bounding_rect(frame, hand_landmarks)
#             landmark_list = calc_landmark_list(frame, hand_landmarks)

#             # Preprocess landmarks and point history (same as before)

#             hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
#             if hand_sign_id == 2:  # Pointing sign
#                 point_history.append(landmark_list[8])  # Index finger coordinates
#             else:
#                 point_history.append([0, 0])

#             # Perform finger gesture recognition (same as before)

#             frame = draw_bounding_rect(True, frame, brect)
#             frame = draw_landmarks(frame, landmark_list)
#             frame = draw_info_text(frame, brect, handedness,
#                                    keypoint_classifier_labels[hand_sign_id],
#                                    point_history_classifier_labels[most_common_fg_id[0][0]])

#             if hand_sign_id != -1 and keypoint_classifier_labels[hand_sign_id] != predictions:
#                 predictions += keypoint_classifier_labels[hand_sign_id] + " "

#     else:
#         point_history.append([0, 0])

#     frame = draw_point_history(frame, point_history)
#     frame = draw_info(frame, cvFpsCalc.get(), 0, -1)

#     st.image(frame, use_column_width=True)  # Display the updated frame

#     if cv.waitKey(1) == 27:  # ESC
#         break

# cap.release()
# cv.destroyAllWindows()

# if predType != "Webcam":
#     st.write("Output sentence:")
#     st.write(predictions)


import cv2 as cv
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from model import KeyPointClassifier, PointHistoryClassifier
from collections import Counter, deque
from utils import CvFpsCalc
import csv
import itertools
import copy
import multiprocessing as mp

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
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

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
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # 正規化
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        if n == 0 or max_value == 0:
            return 0
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

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # 1次元リストに変換
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1:
        # csv_path = 'model/keypoint_classifier/keypoint.csv'
        csv_path = 'model/keypoint_classifier/alphakeypoints.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([24, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmark_point):
    # 接続線
    if len(landmark_point) > 0:
        # 親指
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # 人差指
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # 中指
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # 薬指
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # 小指
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # 手の平
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # キーポイント
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # 外接矩形
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    # for index, point in enumerate(point_history):
    #     if point[0] != 0 and point[1] != 0:
    #         cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
    #                   (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


class HandGestureRecognition(VideoTransformerBase):
    def __init__(self):
        super().__init__()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        self.keypoint_classifier = KeyPointClassifier()
        self.point_history_classifier = PointHistoryClassifier()

        with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            self.keypoint_classifier_labels = csv.reader(f)
            self.keypoint_classifier_labels = [row[0] for row in self.keypoint_classifier_labels]

        with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
            self.point_history_classifier_labels = csv.reader(f)
            self.point_history_classifier_labels = [row[0] for row in self.point_history_classifier_labels]

        self.cvFpsCalc = CvFpsCalc(buffer_len=10)
        self.history_length = 16
        self.point_history = deque(maxlen=self.history_length)
        self.finger_gesture_history = deque(maxlen=self.history_length)
        self.predictions = ""

    

    def transform(self, frame):
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = self.hands.process(frame)
        frame.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            left_hand_landmarks = None
            right_hand_landmarks = None

            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = calc_bounding_rect(frame, hand_landmarks)
                landmark_list = calc_landmark_list(frame, hand_landmarks)

                prl1 = pre_process_landmark(landmark_list[:21])
                prl2 = pre_process_landmark(landmark_list[21:])
                pre_processed_landmark_list = prl1 + prl2

                pre_processed_point_history_list = pre_process_point_history(frame, self.point_history)

                hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # Pointing sign
                    self.point_history.append(landmark_list[8])  # Index finger coordinates
                else:
                    self.point_history.append([0, 0])

                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (self.history_length * 2):
                    finger_gesture_id = self.point_history_classifier(pre_processed_point_history_list)

                self.finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(self.finger_gesture_history).most_common()

                frame = draw_bounding_rect(True, frame, brect)
                frame = draw_landmarks(frame, landmark_list)
                frame = draw_info_text(frame, brect, handedness,
                                       self.keypoint_classifier_labels[hand_sign_id],
                                       self.point_history_classifier_labels[most_common_fg_id[0][0]])

                if hand_sign_id != -1 and self.keypoint_classifier_labels[hand_sign_id] != self.predictions:
                    self.predictions += self.keypoint_classifier_labels[hand_sign_id] + " "

        else:
            self.point_history.append([0, 0])

        frame = draw_point_history(frame, self.point_history)
        frame = draw_info(frame, self.cvFpsCalc.get_fps(), 0, -1)

        return frame

# Initialize CvFpsCalc object
cvFpsCalc = CvFpsCalc(buffer_len=10)

# Title and description
st.title("Hand Gesture Recognition with Streamlit and WebRTC")
st.write("This app performs live hand gesture recognition using Streamlit and WebRTC.")

# Create an instance of HandGestureRecognition class
hand_gesture_recognizer = HandGestureRecognition()

# Display the live webcam stream and apply the HandGestureRecognition processing
webrtc_streamer(key="hand_gesture_recognition", video_transformer_factory=HandGestureRecognition)
