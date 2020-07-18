import cv2
import numpy as np


framewidth = 640
frameheight = 400
cap = cv2.VideoCapture(0)
cap.set(3, framewidth)
cap.set(4,frameheight)




while True:
    success, img = cap.read()

    imgBlur = cv2.GaussianBlur(img, (7,7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Result", img)
    if cv2.waitkey(1) &0xFF == ord('q'):
        break