import mediapipe as mp
import cv2 as cv
import math

camNum = 0
imgHeight = 480
imgWidth = 620
quitKey = 'q'

indexFingerVector = []
normalVector = [0, 300]

# MediaPipe framework for finding Hand Landmarks
drawingModule = mp.solutions.drawing_utils
handsModule = mp.solutions.hands

# OpenCV Webcam Initialization
cap = cv.VideoCapture(camNum)

hand = handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)

def getLandmarkCo(img, landmarkNum):
    
    for handLandmarks in results.multi_hand_landmarks:
        drawingModule.draw_landmarks(img, handLandmarks, handsModule.HAND_CONNECTIONS)

        landmark = handLandmarks.landmark[landmarkNum]

        h, w, _ = img.shape
        cx, cy = int(landmark.x *w), int(landmark.y *h) 

    return cx, cy 

def getLineVector(cx1, cy1, cx2, cy2):
    lineVector = [cx2-cx1, cy2-cy1]
    return lineVector

def getLength(vector):
    length = math.sqrt(pow(vector[0], 2) + pow(vector[1], 2))
    return length

def getDotProduct(vector1, vector2):
    dotProduct = (vector1[0] * vector2[0]) + (vector1[1] * vector2[1])
    return dotProduct

def getAngle(vector1, vector2):
    length1 = getLength(vector1)
    length2 = getLength(vector2)
    dotProduct = getDotProduct(vector1, vector2)
    angle = math.acos( dotProduct / (length1 * length2))
    angle = math.degrees(angle)
    return angle

while True:
    _, img = cap.read()
    img = cv.resize(img, (imgWidth, imgHeight))
    key = cv.waitKey(1) & 0xFF
    
    results = hand.process(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    if  results.multi_hand_landmarks != None:
        # Gather LandMark Coordinates 
        indexTipCX, indexTipCY = getLandmarkCo(img, 8)
        palmBaseCX, palmBaseCY = getLandmarkCo(img, 0)
       
        # Create Line Vector between Landmarks
        lineVector = getLineVector(indexTipCX, indexTipCY, palmBaseCX, palmBaseCY)
        
        # Gather Angle Between Line Vector and Normal Vector
        angle = getAngle(normalVector, lineVector)

        # Landmark Coordinates  
        cv.putText(img, f"Index Finger Tip X-Coord: {indexTipCX}", (80, 390), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        
        cv.putText(img, f"Index Finger Tip Y-Coord: {indexTipCY}", (80, 410), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        cv.putText(img, f"Palm Base X-Coord: {palmBaseCX}", (120, 430), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        cv.putText(img, f"Palm Base Y-Coord: {palmBaseCY}", (120, 450), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        
        # Line Vectors
        cv.putText(img, f"Line Vector: {lineVector}", (40, 370), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
         
        # Angle
        cv.putText(img, f"Angle: {angle}", (40,270), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        

        print(f"Index Finger Tip X-Coord:{indexTipCX}\nIndex Finger Tip Y-Coord:{indexTipCY}")

        print(f"Palm Base X-Coord:{palmBaseCX}\nPalm Base Y-Coord{palmBaseCY}")


    cv.imshow("HandLandmark Coordinate", img)
    if key == ord(quitKey):
       break

cap.release()
cv.destroyAllWindows()



















                                          
