import mediapipe as mp
import cv2 as cv
import math

camNum = 0
imgHeight = 480
imgWidth = 620
quitKey = 'q'

indexTipLandmark = 8
middleTipLandmark = 12

# Mediapipe framework for finding Hand Landmarks
drawingModule = mp.solutions.drawing_utils
handsModule = mp.solutions.hands

# OpenCV Webcam Initialization
cap = cv.VideoCapture(camNum)

hand = handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)

def getLandmarkCo(img, landmarkNum):
    for handLandmarks in results.multi_hand_landmarks:
        drawingModule.draw_landmarks(img, handLandmarks, handsModule.HAND_CONNECTIONS)  
        
        landmark = handLandmarks.landmark[landmarkNum]

        h, w, _= img.shape
        cx, cy = int(landmark.x *w), int(landmark.y*h)

    return cx, cy

def getLineVector(cx1, cy1, cx2, cy2):
    lineVector = [cx2-cx1, cy2-cy1]
    return lineVector

def getLength(vector):
    length = math.sqrt(pow(vector[0],2) + pow(vector[1],2))
    return length

while True:
    _, img = cap.read()
    img = cv.resize(img, (imgWidth, imgHeight))
    key = cv.waitKey(1) & 0xFF

    results = hand.process(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    if results.multi_hand_landmarks != None:
        indexTipCX, indexTipCY = getLandmarkCo(img, indexTipLandmark)
        middleTipCX, middleTipCY = getLandmarkCo(img, middleTipLandmark)

        lineVector = getLineVector(indexTipCX, indexTipCY,middleTipCX, middleTipCY)
        length = getLength(lineVector)
        
        # Landmark Coordinates  
        cv.putText(img, f"Index Finger Tip X-Coord: {indexTipCX}", (80, 390), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        
        cv.putText(img, f"Index Finger Tip Y-Coord: {indexTipCY}", (80, 410), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        cv.putText(img, f"Middle Finger Tip X-Coord: {middleTipCX}", (120, 430), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        cv.putText(img, f"Middle Finger Tip Y-Coord: {middleTipCY}", (120, 450), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        
        # Length of Line Vectors
        cv.putText(img, f"Length of Line Vector: {length}", (40, 370), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        
                
        
    cv.imshow("HandLandmark Coordinate", img)
    if key == ord(quitKey):
       break

cap.release()
cv.destroyAllWindows()

                        



