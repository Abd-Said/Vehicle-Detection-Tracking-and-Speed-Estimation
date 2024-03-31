import cv2
import numpy as np
import time
from tracker import *
import math

start_position = None
cap = cv2.VideoCapture("C:/Users/user/Desktop/yolculuk.mp4") #your video adress here. You can use webcam with 0.
time.sleep(2)

tracker = EuclideanDistTracker()
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=100)

while True:
    ret,frame = cap.read()

    height , width, _ = frame.shape

    roi = frame[400:2000,1100:2000] #choose this for your best detection area (this way decreases flops)

    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 55,255,cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections=[]
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 300:
            x,y,w,h = cv2.boundingRect(cnt)

            detections.append([x, y, w, h])

    boxes_ids =  tracker.update(detections)
    for box_id in boxes_ids:
        x,y,w,h,id=box_id
        cv2.putText(roi, str(id), (x,y-15), cv2.FONT_HERSHEY_PLAIN, 1 ,(255,0,0),2)

        current_position = [x + w// 2 , y + h//2]

        if start_position is None:
            start_position = current_position
            start_time = time.time()
        else:
            elapsed_time = time.time() - start_time

            if elapsed_time != 0:
                speedx = abs((current_position[0] - start_position[0]) / elapsed_time)
                speedy = abs((current_position[1] - start_position[1]) / elapsed_time)
                speed = math.sqrt(speedx**2 + speedy**2)
              
            if id % 3 == 1:
                cv2.putText(frame, f'Speed{str(id)}:{round(speed, 2)} pixel/s', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif id%3==2:
                cv2.putText(frame, f'Speed{str(id)}:{round(speed, 2)} pixel/s', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2)
            else:
                cv2.putText(frame, f'Speed{str(id)}:{round(speed, 2)} pixel/s', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2)

        key = cv2.waitKey(30)
        if key == 27:
          break

    cv2.imshow("roi",roi)
    cv2.imshow('webcam', frame)
    cv2.imshow("maske",mask)

    if cv2.waitKey(30)&0xFF == ord('q'):
         break
      
cap.release()
cv2.destroyAllWindows()
