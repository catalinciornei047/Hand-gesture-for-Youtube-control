import csv
import copy
import argparse
import itertools
import threading
import pyautogui
import time
from concurrent.futures import ProcessPoolExecutor
import cv2 as cv
import numpy as np
import mediapipe as mp
from utils import CvFpsCalc
from model import KeyPointClassifier

executor = ProcessPoolExecutor(max_workers=2)
action_registered = {
    'volume up': False,
    'volume down': False,
    'forward': False,
    'backward': False,
    'finger-back': False,
    'finger-next': False,
    'ok-nothing': False,
    'play': False,
    'play-nothing': False,
    'fourfingers': False,

}

cap = {}
hands = {}

def get_args():
    # Video camera settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=1280)
    parser.add_argument("--height", help='cap height', type=int, default=720)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",help='min_detection_confidence',type=float,default=0.8)
    parser.add_argument("--min_tracking_confidence",help='min_tracking_confidence',type=int,default=0.5)
    args = parser.parse_args()
    return args

def action_hand_sign(hand_sign_label):
    global action_registered
    hand_sign_label = hand_sign_label.lower()
    # Reset all actions to False
    for key in action_registered:
        if key != hand_sign_label:
            action_registered[key] = False

    if not action_registered[hand_sign_label]:
        if hand_sign_label == 'play-nothing':
            pass
        elif hand_sign_label == 'ok-nothing':
            pass
        elif hand_sign_label == 'finger-back':
            pyautogui.hotkey('alt','left', presses=1)
            time.sleep(0.2)
        elif hand_sign_label == 'finger-next':
            pyautogui.hotkey('alt', 'right', presses=1)
            time.sleep(0.2)
        elif hand_sign_label == 'forward':
            pyautogui.press('right')
            time.sleep(0.2)
        elif hand_sign_label == 'backward':
            pyautogui.press('left')
            time.sleep(0.2)
        elif hand_sign_label == 'volume up':
            pyautogui.press('up')
            time.sleep(0.2)
        elif hand_sign_label == 'volume down':
            pyautogui.press('down')
            time.sleep(0.2)
        elif hand_sign_label == 'play':
            pyautogui.press('k', presses=1)
            time.sleep(0.2)
        elif hand_sign_label == 'fourfingers':
            pyautogui.press('m')
            time.sleep(0.2)
        action_registered[hand_sign_label] = True

def video_streamer_control(keypoint_classifier_labels, mode):
    # Video control functions for hand gesture recognition
    global cap, hands
    keypoint_classifier = KeyPointClassifier()
    timp_trecut = time.time()
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    use_brect = True
    fpsSet = 0.5
    while True:
        if (hasattr(cap, "read")):
            fps = cvFpsCalc.get()
            # Process Key (ESC: end) #################################################
            key = cv.waitKey(10)
            if key == 27:  # ESC
                break
            number, mode = select_mode(key, mode)
            # Camera capture #####################################################
            ret, image = cap.read()
            if not ret:
                break
            image = cv.flip(image, 1)  # Mirror display
            debug_image = copy.deepcopy(image)
            # Detection implementation #############################################################
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            #  ####################################################################
            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                      results.multi_handedness):
                    # Bounding box calculation
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    # Landmark calculation
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(
                        landmark_list)
                    # Write to the dataset file
                    logging_csv(number, mode, pre_processed_landmark_list)
                    # Hand sign classification
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    # Hand Sign Action
                    hand_sign_label = keypoint_classifier_labels[hand_sign_id]
                    if (time.time() > timp_trecut + fpsSet):
                        timp_trecut = time.time()
                        executor.submit(action_hand_sign, hand_sign_label)
                    # Drawing part
                    debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = draw_info_text(
                        debug_image,
                        brect,
                        handedness,
                        keypoint_classifier_labels[hand_sign_id],
                    )
            debug_image = draw_info(debug_image, fps, mode, number)
            # Screen reflection #############################################################
            cv.imshow('Hand Gesture Youtube Control ', debug_image)

def video_control():
    # Prepare video capture and the detection model
    global cap, hands
    # Argument parsing #################################################################
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

def main():
    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    mode = 0
    # Separate flow of execution by threads t1, t2
    t1 = threading.Thread(target=video_streamer_control, args=(keypoint_classifier_labels, mode))
    t2 = threading.Thread(target=video_control)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    cap.release()
    cv.destroyAllWindows()

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    return number, mode

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
    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))
    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Define the connections between landmarks
        connections = [
            (2, 3), (3, 4),  # Thumb
            (5, 6), (6, 7), (7, 8),  # Index finger
            (9, 10), (10, 11), (11, 12),  # Middle finger
            (13, 14), (14, 15), (15, 16),  # Ring finger
            (17, 18), (18, 19), (19, 20),  # Little finger
            (0, 1), (1, 2), (2, 5), (5, 9),  # Palm
            (9, 13), (13, 17), (17, 0)
        ]
        # Draw landmark connections
        for connection in connections:
            cv.line(image, tuple(landmark_point[connection[0]]), tuple(landmark_point[connection[1]]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[connection[0]]), tuple(landmark_point[connection[1]]),
                    (51, 255, 51), 2)
    # Draw key points
    for index, landmark in enumerate(landmark_point):
        color = (255, 255, 255)
        border_color = (0, 0, 0)
        radius = 5
        if index in [4, 8, 12, 16, 20]:
            radius = 8
        cv.circle(image, (landmark[0], landmark[1]), radius, color, -1)
        cv.circle(image, (landmark[0], landmark[1]), radius, border_color, 1)
    return image

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)
    return image

def draw_info_text(image, brect, handedness, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)
    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image

def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)
    mode_string = ['Key Point Classifier']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image

if __name__ == '__main__':
    main()
