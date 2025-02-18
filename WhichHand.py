# *****************************************************************************
# ***************************  Python Source Code  ****************************
# *****************************************************************************
#
#   DESIGNER NAME:  Mario Solis
#
#       FILE NAME:  WhichHand.py
#
# DESCRIPTION
#   This code uses mediapipe to detect the landmarks of the hand through the
#   handsModule then opencv to draw a box around the detected hand.
#
#   This code was written building off HandSpotters observations and reading
#   the mediapipe github repo.(mediapipe/mediapipe/python/solutios/hands.py)
#
#   Im looking to build my familiarity with this library. Slowely working my 
#   way up to specifying the location of each landmark. To use this information
#   to improve the gesture drone project. 
#
#   A few ideas I've gathered so far is in observing what kind of math 
#   operations can be used to detect when the landmark positionings are
#   within a specific gesture formation relative to each other.
#  
import mediapipe as mp
import cv2 as cv

camNum = 0
imgHeight = 480
imgWidth = 640
quiteKey = 'q'

# MediaPipe framework for Finding Hand Landmarks
drawingModule = mp.solutions.drawing_utils
handsModule = mp.solutions.hands

# OpenCV Webcam VideoStream
cap = cv.VideoCapture(camNum)

hand = handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)

while True:
    # Live Webcam Feed Block
    sucess, img = cap.read()
    img = cv.resize(img, (imgWidth, imgHeight))
   
    # Using cv.flip I was able to mirror the webcam feed and correct the
    # inconsistency around the mix up of left and right hand detection.
    img = cv.flip(img, 2) 
    key = cv.waitKey(1) & 0xFF

    # Process Hand Framework
    results = hand.process(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    if results.multi_hand_landmarks != None:
        for handLandmarks in results.multi_hand_landmarks:
            drawingModule.draw_landmarks(img, handLandmarks, handsModule.HAND_CONNECTIONS)
        print(results.multi_handedness)

    cv.imshow("HandLandmarks", img)
    if key == ord(quiteKey):
        break

cap.release()
cv.destroyAllWindows()
