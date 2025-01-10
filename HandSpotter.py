


import mediapipe as mp
import cv2 as cv

camNum = 0
imgHeight = 480
imgWidth = 640
quiteKey = 'q'

# MediaPipe framework for Finding Hand Landmarks
drawingModule = mp.solutions.drawing_utils
handsModule = mp.solutions.hands

# Hand Box
#handBox = drawingModule.draw_landmarks(img, handLandmarks, handsModule.HAND_CONNECTIONS)


# OpenCV Webcam VideoStream
cap = cv.VideoCapture(camNum)

# Setup HandModule Settings
with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1) as hands:

    while True:
        # Live Webcam Feed Block
        sucess, img = cap.read()
        img = cv.resize(img, (imgWidth,imgHeight))
        key = cv.waitKey(1) & 0xFF
        
        # Process Hand Framework
        results = hands.process(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        if results.multi_hand_landmarks != None:
            for handLandmarks in results.multi_hand_landmarks:
                drawingModule.draw_landmarks(img, handLandmarks, handsModule.HAND_CONNECTIONS)
                
                h, w, _ = img.shape
                xMin, yMin = w, h
                xMax, yMax = 0, 0
                for lm in handLandmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    xMin, yMin = min(xMin, x), min(yMin, y)
                    xMax, yMax = max(xMax, x), max(yMax, y)

                cv.rectangle(img, (xMin, yMin), (xMax, yMax), (0, 0, 255), 2)

        cv.imshow("HandSpotter", img)
        if key == ord(quiteKey):
            break
        
cap.release()
cv.destroyAllWindows()
    


