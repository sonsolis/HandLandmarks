import mediapipe as mp
import cv2 as cv
import math
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

camNum = 0
imgHeight = 420
imgWidth = 680
quitKey = 'q'

# Initialize Mediapipe Hand Recognition Tools
drawingModule = mp.solutions.drawing_utils
handsModule = mp.solutions.hands 

# Initialize Webcam
cap = cv.VideoCapture(camNum)

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = vision.GestureRecognizer
GestureRecognizerOptions = vision.GestureRecognizerOptions
VisionRunningMode = vision.RunningMode

# Live Stream Running Mode Callback 
def gesture_callback(result, image, timestamp_ms):
    if result.gestures:
        if not hasattr(gesture_callback, "top_gesture"):

            gesture_callback.top_gesture = result.gesture[0][0]
            print(f"Recognized gesture: {top_gesture.category_name} "
                  f"Score: {top_gesture.score:.2f} at time: {timestamp_ms}ms")

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path="gesture_recognizer.task"),running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=gesture_callback,)

# Configure Handlandmark Detection Settings
hand = handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)

def getLandmarkCo(img, landmarkNum):
    for handLandmarks in results.multi_hand_landmarks:
        drawingModule.draw_landmarks(img, handLandmarks, handsModule.HAND_CONNECTIONS)

        landmark = handLandmarks.landmark[landmarkNum]

        h, w, _ = img.shape
        cx, cy = int(landmark.x *w), int(landmark.y *h)

    return cx, cy

# Initialize Gesture Recognizer
with GestureRecognizer.create_from_options(options) as recognizer:
    while True:
        _, img = cap.read()
        key = cv.waitKey(1) & 0xFF
        rgb_frame = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # Display Gesture Name
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        current_time_ms = int(time.time() * 1000)
        recognizer.recognize_async(mp_image, current_time_ms)
        cv.putText(img, gesture_callback.top_gesture, (50,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

        cv.imshow("Gesture Recognition", frame)

        if key == ord(quitKey):
            break

