import cv2
import numpy as np
from PIL import Image

def get_limits(color):

    c = np.uint8([[color]])  # here insert the bgr values which you want to convert to hsv
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    lowerLimit = hsvC[0][0][0] - 10, 100, 100
    upperLimit = hsvC[0][0][0] + 10, 255, 255

    lowerLimit = np.array(lowerLimit, dtype=np.uint8)
    upperLimit = np.array(upperLimit, dtype=np.uint8)

    return lowerLimit, upperLimit
  
blue = [255, 0, 0]  # Red in BGR colorspace

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowerLimit, upperLimit = get_limits(color=blue)

    mask  = cv2.inRange(hsvImage, lowerLimit,upperLimit)
    mask_ = Image.fromarray(mask)
    boundingBox = mask_.getbbox() 
    if boundingBox is not None:
        x1, y1, x2, y2 = boundingBox
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
        
    out = cv2.flip(frame, 2)
    cv2.imshow('Frame', out)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
