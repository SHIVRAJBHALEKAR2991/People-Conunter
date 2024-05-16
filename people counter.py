import cv2 as cv
from ultralytics import YOLO
import numpy as np
import math
import cvzone
from sort import *

model =YOLO('yolov8l.pt')
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

cap=cv.VideoCapture('Videos/people.mp4')
cap.set(3,640)
cap.set(4,480)

maskup=cv.imread('maskcombined.png')

# tracker
tracker = Sort(max_age=20,min_hits=1 ,iou_threshold=0.1)
limitsup=[180,400,400,400]
limitsdown=[480,400,700,400]
totalup=[]
totaldown=[]

while True:
    success,img=cap.read()

    imgroiup=cv.bitwise_and(img,maskup)
    result=model(imgroiup,stream=True)
    detections = np.empty((0, 5))
    for r in result:

        boxes = r.boxes
        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv.rectangle(img,(x1,y1),(x2,y2),(0,0,255),thickness=2)
            # bounding box
            w, h = x2 - x1, y2 - y1

            # confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # print(conf)

            # class
            cls = int(box.cls[0])
            currentclass = classNames[cls]

            if((currentclass=='person') and conf>0.3):
                currentarray=np.array([x1,y1,x2,y2,conf])
                detections=np.vstack((detections,currentarray))
    resultstravker = tracker.update(detections)

    cv.line(img,(limitsup[0],limitsup[1]),(limitsup[2],limitsup[3]),(0,0,255),thickness=2)
    cv.line(img, (limitsdown[0], limitsdown[1]), (limitsdown[2], limitsdown[3]), (255, 0,255 ), thickness=2)

    for res in resultstravker:

        x1,y1,x2,y2,id= res
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        w,h=x2-x1,y2-y1
        cx, cy = x1 + w // 2, y1 + h // 2

        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=0.8, thickness=1, offset=5)
        cv.circle(img, (cx, cy), radius=2, color=(0, 255, 0), thickness=cv.FILLED)

        if (limitsup[0]<cx<limitsup[2]) and (limitsup[1]-20<cy<limitsup[3]+20):
            if totalup.count(id)==0:
                totalup.append(id)

        if (limitsdown[0]<cx<limitsdown[2]) and (limitsdown[1]-20<cy<limitsdown[3]+20):
            if totaldown.count(id)==0:
                totaldown.append(id)



    cvzone.putTextRect(img,f'up:{len(totalup)}',pos=(50,50))
    cvzone.putTextRect(img, f'down:{len(totaldown)}', pos=(850, 50))
    cv.imshow('images', img)
    cv.waitKey(1)



