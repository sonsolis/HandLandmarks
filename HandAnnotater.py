import mediapipe as mp
import cv2 as cv
import os

filePath = "./G1"

fileImgs = os.listdir(filePath)

# MediaPipe framework for Finding Hand Landmarks
drawingModule = mp.solutions.drawing_utils
handsModule = mp.solutions.hands

hand = handsModule.Hands(static_image_mode=True, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1) 


for img in fileImgs:
    
    imgPath = f"{filePath}/{img}"
    rawImg = cv.imread(imgPath)
    
    if rawImg is None:
        print("Error Img Skipped")

    else:
        rawImg = cv.resize(rawImg, (640, 480))
        results = hand.process(cv.cvtColor(rawImg, cv.COLOR_BGR2RGB))
        if results.multi_hand_landmarks != None:
            for handLandmarks in results.multi_hand_landmarks:
                drawingModule.draw_landmarks(rawImg, handLandmarks, handsModule.HAND_CONNECTIONS)
                h, w, _ = rawImg.shape
                xMin, yMin = w, h
                xMax, yMax = 0, 0
                for lm in handLandmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    xMin, yMin = min(xMin, x), min(yMin, y)
                    xMax, yMax = max(xMax, x), max(yMax, y)

                cv.rectangle(rawImg, (xMin, yMin), (xMax, yMax), (255, 0, 255), 2)

            os.makedirs(f"./Annotated/G1/", exist_ok=True)
            cv.imwrite(f"./Annotated/G1/{img}", rawImg)


