import mediapipe as mp
import cv2 as cv
import time
import math
import threading
import queue

# Gesture Recognizer Model Imports
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

###########################################################

# Gesture Recognizer Model Setup Variables
baseOptions = mp.tasks.BaseOptions
gestureRecognizer = vision.GestureRecognizer
gestureRecognizerOptions = vision.GestureRecognizerOptions
visionRunningMode = vision.RunningMode

# Shared Global Variable for Recognized Gestures
latestGestureInfo = None
gestureLock = threading.Lock()
gestureQueue = queue.Queue()

# Gesture Recognizer Callback Function
def gestureCallback(result, image, timestamp_ms):
    global latestGestureInfo
    if result.gestures:
        topGesture = result.gestures[0][0]
        gestureData = {
                "gestureName": topGesture.category_name,
                "score": topGesture.score,
                "timestamp": timestamp_ms
                }
        gestureQueue.put(gestureData)
        with gestureLock:
            latestGestureInfo = {
                        "gestureName": topGesture.category_name,
                        "score": topGesture.score,
                        "timestamp": timestamp_ms
                    }
        
            # print(f"Recognized gesture: {topGesture.category_name} "
            #      f"Score: {topGesture.score:.2f} at time: {timestamp_ms}ms")

# Gesture Recognizer Setup 
options = gestureRecognizerOptions(
        base_options=baseOptions(model_asset_path="gesture_recognizer.task"),
        running_mode=visionRunningMode.LIVE_STREAM,
        result_callback=gestureCallback,)

###########################################################

# OpenCV Setup Variables
camNum = 0
imgHeight = 480
imgWidth = 640
quitKey = 'q'

# OpenCV Webcam Setup
cap = cv.VideoCapture(camNum)

############################################################

# Hand Landmark Variables
indexFingerTipLandmark = 8
wristLandmark = 0
middleFingerTipLandmark = 12

# Hand Landmark Drawing Module
drawingModule = mp.solutions.drawing_utils
handsModule = mp.solutions.hands

# Hand Landmark Setup
hand = handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)

############################################################

# Hand Landmark Functions
def getLandmarkCo(img, landmarkNum):
    for handLandmarks in results.multi_hand_landmarks:
        # drawingModule.draw_landmarks(img, handLandmarks, handsModule.HAND_CONNECTIONS)
        landmark = handLandmarks.landmark[landmarkNum]

        h, w, _ = img.shape
        cx, cy = int(landmark.x *w), int(landmark.y *h)

    return cx, cy

############################################################

# Math Variables
normalVector = [300,0]

############################################################

# Math Variables
angle = 0
tuckLength = 0

# Math Functions
def getLineVector(cx1, cy1, cx2, cy2):
    lineVector = [cx2-cx1, cy2-cy1]
    return lineVector

def getLength(vector):
    length = math.sqrt(pow(vector[0], 2) + pow(vector[1],2))
    return length

def getDotProduct(vector1, vector2):
    dotProduct = (vector1[0] * vector2[0]) + (vector1[1] * vector2[1])
    return dotProduct

def getCrossProduct(vector1, vector2):
    crossProduct = (vector1[0] * vector2[1])-(vector1[1] * vector2[0])
    return crossProduct

def getAngle(vector1, vector2):
    dotProduct = getDotProduct(vector1, vector2)
    crossProduct = getCrossProduct(vector1, vector2)
    angle = math.atan2(crossProduct, dotProduct)
    angle = math.degrees(angle)
    if angle < 0:
        angle = angle + 360
    return angle 


############################################################

# Gesture Control Functions
def gestureCaller(gesture):
    match gesture:
        case "Open_Palm":
            time.sleep(1)
        case "Closed_Fist":
            time.sleep(1)
        case "Thumb_Up":
            print("Take Off")
        case "Thumb_Down":
            print("Land")
        case _:
            pass

def angleCaller(angle, tuckLength):
    
    match tuckLength:
        case _ if tuckLength >= 50:

            match angle:
                case _ if 0 <= angle <= 45 or 320 <= angle <= 0:
                    print("left")
                case _ if 45 <= angle <= 135:
                    print("forward")
                case _ if 135 <= angle <= 225:
                    print("right")
                case _ if 225 <= angle <= 320:
                    print("backward")
                case _:
                    pass
        case _ if tuckLength <= 50:
            match angle:
                case _ if 0 <= angle <= 45 or 320 <= angle <= 0:
                    print("rotae left")
                case _ if 45 <= angle <= 135:
                    print("up")
                case _ if 135 <= angle <= 225:
                    print("rotate right")
                case _ if 225 <= angle <= 320:
                    print("down")
                case _:
                    pass
        case _:
            pass

with gestureRecognizer.create_from_options(options) as recognizer:

    while cap.isOpened():
        _, img = cap.read()
        img = cv.resize(img, (imgWidth, imgHeight))
        img = cv.flip(img, 3)
        key = cv.waitKey(1) & 0xFF

        results = hand.process(cv.cvtColor(img, cv.COLOR_BGR2RGB))

        if  results.multi_hand_landmarks != None:
            # Gather LandMark Coordinates 
            indexTipCX, indexTipCY = getLandmarkCo(img, indexFingerTipLandmark)
            palmBaseCX, palmBaseCY = getLandmarkCo(img, wristLandmark)
            
            middleTipCX, middleTipCY = getLandmarkCo(img, middleFingerTipLandmark)
            # Create Line Vector between Landmarks
            lineVector = getLineVector(indexTipCX, indexTipCY, palmBaseCX, palmBaseCY)
            
            tuckVector = getLineVector(indexTipCX, indexTipCY, middleTipCX, middleTipCY)

            tuckLength = getLength(tuckVector)

            # Gather Angle Between Line Vector and Normal Vector
            angle = getAngle(normalVector, lineVector)
            
            # Landmark Coordinates  
#             cv.putText(img, f"Index Finger Tip X-Coord: {indexTipCX}", (80, 390), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
#             
#             cv.putText(img, f"Index Finger Tip Y-Coord: {indexTipCY}", (80, 410), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
# 
#             cv.putText(img, f"Palm Base X-Coord: {palmBaseCX}", (120, 430), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
# 
#             cv.putText(img, f"Palm Base Y-Coord: {palmBaseCY}", (120, 450), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
#             
#             # Line Vectors
#             cv.putText(img, f"Line Vector: {lineVector}", (40, 370), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
            
#             cv.putText(img, f"Tuck Vector: {tuckLength}", (40, 330), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
#              
#             # Angle
#             cv.putText(img, f"Angle: {angle}", (40,270), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
            
#             while not gestureQueue.empty():
#                 gestureData = gestureQueue.get()
#                 # Detected Gesture
#                 cv.putText(img, "Detected Gesture: " + latestGestureInfo["gestureName"], (40, 290), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
# 
            #with gestureLock:
             #   if latestGestureInfo is not None:

                    # Detected Gesture
              #      cv.putText(img, "Detected Gesture: " + latestGestureInfo["gestureName"], (40, 290), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
         
        if not _:
            break
        
        rgbImg = cv.cvtColor(img,  cv.COLOR_BGR2RGB)
        mpImg = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgbImg)
        current_time_ms = int(time.time() *1000)
        recognizer.recognize_async(mpImg, current_time_ms)

        if not gestureQueue.empty(): 
            gestureData = gestureQueue.get()
            gestureCaller(latestGestureInfo["gestureName"])
        else:    
            angleCaller(angle, tuckLength)


        cv.imshow("Live Stream Mode", img)
        if key == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
