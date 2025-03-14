import VectorAngle as vec 
import mediapipe as mp
import cv2 as cv
import datetime 
import time
import math
import queue

# Tello Drone Imports
from djitellopy import tello

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

# Gesture Recognizer Setup 
options = gestureRecognizerOptions(
        base_options=baseOptions(model_asset_path="gesture_recognizer.task"),
        running_mode=visionRunningMode.LIVE_STREAM,
        result_callback=gestureCallback,)

###########################################################

# Tello Drone Variables
me = tello.Tello()
me.connect()
print(me.get_battery())

# Tello Drone Functions
    
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
angle = 0
tuckLength = 0

############################################################

# Gesture Control Variables
vals = [0,0,0,0]
controlDrone = False

# Gesture Control Functions
def gestureCaller(gesture):
    match gesture:
        case "Open_Palm":
            print("Hold")
            me.send_rc_control(0, 0, 0, 0)
        case "Thumb_Up":
            print("Take Off")
            if not me.is_flying:
                me.takeoff()
        case "Thumb_Down":
            print("Land")
            if me.is_flying:
                me.land()
        case _:
            pass

def angleCaller(angle, tuckLength):
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 25

    match tuckLength:
        case _ if tuckLength >= 50:
            match angle:
                case _ if 0 <= angle <= 45 or 320 <= angle <= 0:
                    print("left")
                    lr = -speed
                case _ if 45 <= angle <= 135:
                    print("forward")
                    fb = speed
                case _ if 135 <= angle <= 225:
                    print("right")
                    lr = speed
                case _ if 225 <= angle <= 320:
                    print("backward")
                    fb = -speed
                case _:
                    pass

        case _ if tuckLength <= 50:
            match angle:
                case _ if 0 <= angle <= 45 or 320 <= angle <= 0:
                    print("rotate left")
                    yv = speed
                case _ if 45 <= angle <= 135:
                    print("up")
                    ud = speed
                case _ if 135 <= angle <= 225:
                    print("rotate right")
                    yv = -speed
                case _ if 225 <= angle <= 320:
                    print("down")
                    ud = -speed
                case _:
                    pass
        case _:
            pass

    return [lr, fb, ud, yv]

############################################################

# Cool Down Variables
addOneSecond = datetime.datetime.now() + datetime.timedelta(seconds=1) 
newCycle = datetime.datetime.now() + datetime.timedelta(seconds=1) 
countDown = 0
coolDownOver = True

############################################################

# Main Method

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
            lineVector = vec.getLineVector(indexTipCX, indexTipCY, palmBaseCX, palmBaseCY)

            tuckVector = vec.getLineVector(indexTipCX, indexTipCY, middleTipCX, middleTipCY) 

            tuckLength = vec.getLength(tuckVector)
            # Gather Angle Between Line Vector and Normal Vector
            angle = vec.getAngle(normalVector, lineVector)

            # Landmark Coordinates  
#             cv.putText(img, f"Index Finger Tip X-Coord: {indexTipCX}", (80, 390), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

#             
#             cv.putText(img, f"Index Finger Tip Y-Coord: {indexTipCY}", (80, 410), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
#k
#             cv.putText(img, f"Palm Base X-Coord: {palmBaseCX}", (120, 430), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
# 
#             cv.putText(img, f"Palm Base Y-Coord: {palmBaseCY}", (120, 450), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
#             
#             # Line Vectors
#             cv.putText(img, f"Line Vector: {lineVector}", (40, 370), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

#             cv.putText(img, f"Tuck Vector: {tuckLength}", (40, 330), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
#              

        # Angle
#        cv.putText(img, f"Angle: {angle}", (40,270), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        if not _:
            break

        rgbImg = cv.cvtColor(img,  cv.COLOR_BGR2RGB)
        mpImg = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgbImg)
        current_time_ms = int(time.time() *1000)
        recognizer.recognize_async(mpImg, current_time_ms)

        while not gestureQueue.empty():
            gestureData = gestureQueue.get()
#            print("loop")
            if gestureData["gestureName"] in ["Open_Palm","Thumb_Up","Thumb_Down"]: 
                # print("Gesture Detected")
                gestureCaller(gestureData["gestureName"])
                current = datetime.datetime.now()
                addOneSecond = current+datetime.timedelta(seconds=1)
                newCycle = current + datetime.timedelta(seconds=1)
                coolDownOver = False
                controlDrone = True
            
            if gestureData["gestureName"] in ["None", "Pointing_Up"]:
                # print("Break Condition")
                gestureQueue = queue.Queue() 
                break

        if datetime.datetime.now() > addOneSecond and countDown != 0:
            # print("loop 1")
            countDown -= 1
            countCurrent = datetime.datetime.now()
            addOneSecond = countCurrent + datetime.timedelta(seconds= 1)

        if datetime.datetime.now() > newCycle:
            # print("loop 2")
            coolDownOver = True

        if coolDownOver == True:
            vals = angleCaller(angle, tuckLength)
            # angleCaller(angle, tuckLength)

        if controlDrone == True:
            me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
            pass

        cv.imshow("Live Stream Mode", img)
        if key == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
