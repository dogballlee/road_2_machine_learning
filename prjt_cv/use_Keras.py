import cv2
import numpy as np

things_dtct_path = r'C:\\Users\\Administrator\\AppData\\Local\\Programs\\Python\\Python37\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml'      #预先训练好的特征池


framewidth = 640
frameheight = 400
cap = cv2.VideoCapture(0)
cap.set(3, framewidth)
cap.set(4, frameheight)

def discern(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cap = cv2.CascadeClassifier(things_dtct_path)
    faceRects = cap.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(200, 200))
    if len(faceRects):
        for faceRect in faceRects:
            x, y, w, h = faceRect
            cv2.rectangle(img, (x, y), (x + h, y + w), (0, 255, 0), 2)  # 框出人脸
    cv2.imshow("Image", img)


while True:
    success, img = cap.read()
    discern(img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break