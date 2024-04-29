
import cv2 
import pandas as pd 
from ultralytics import YOLO 
from tracker import * 
import cvzone 
import numpy as np 
 
model = YOLO('yolov8s.pt') 
 
def RGB(event, x, y, flags, param): 
    if event == cv2.EVENT_MOUSEMOVE: 
        point = [x, y] 
        print(point) 
 
cv2.namedWindow('RGB') 
cv2.setMouseCallback('RGB', RGB) 
cap = cv2.VideoCapture('vidp.mp4') 
 
my_file = open("coco.txt", "r") 
data = my_file.read() 
class_list = data.split("\n")  
#print(class_list) 
 
count = 0 
 
area1 = [(494,289),(505,499),(578,496),(530,292)] 
area2 = [(548,290),(600,496),(637,493),(574,288)] 
 
while True:     
    ret, frame = cap.read() 
    if not ret: 
        break 
 
    frame = cv2.resize(frame, (1020,500)) 
 
    results = model.predict(frame) 
    a = results[0].boxes.data 
    px = pd.DataFrame(a).astype("float") 
 
    for index, row in px.iterrows(): 
        x1 = int(row[0]) 
        y1 = int(row[1]) 
        x2 = int(row[2]) 
        y2 = int(row[3]) 
        d = int(row[5]) 
        confidence = row[4]  # Confidence score 
 
        c = class_list[d] 
        if 'person' in c: 
            cv2.rectangle(frame, (x1,y1), (x2,y2), (225,255,255), 2) 
            cvzone.putTextRect(frame, f'{c} {confidence:.2f}', (x1,y1), 1, 1) 
 
    cv2.imshow("RGB", frame) 
    if cv2.waitKey(1) & 0xFF == 27: 
        break 
 
cap.release() 
cv2.destroyAllWindows() 
 
 
 
 
 
 
 
 
 
 
# # accuracy count 
# # Persons count in Crowd  
# import cv2 
# import pandas as pd 
# from ultralytics import YOLO 
# from tracker import * 
# import cvzone 
# import numpy as np 
# import random 
# import time 
 
# model = YOLO('yolov8s.pt') 
 
# def RGB(event, x, y, flags, param): 
#     if event == cv2.EVENT_MOUSEMOVE: 
#         point = [x, y] 
#         print(point) 
 
# cv2.namedWindow('RGB') 
# cv2.setMouseCallback('RGB', RGB) 
# cap = cv2.VideoCapture('vidp.mp4') 
 
# my_file = open("coco.txt", "r") 
# data = my_file.read() 
# class_list = data.split("\n") 
# count = 0 
 
# tracker = Tracker() 
 
# area1 = [(494, 289), (505, 499), (578, 496), (530, 292)] 
# area2 = [(548, 290), (600, 496), (637, 493), (574, 288)] 
 
# person_counter = 0    
# counted_persons = set()    
# total_detected_persons = 0  # Initialize the total detected persons counter 
# start_time = time.time() 
 
# while True: 
#     ret, frame = cap.read() 
#     if not ret: 
#         break 
 
#     frame = cv2.resize(frame, (1020, 500)) 
 
#     results = model.predict(frame) 
#     a = results[0].boxes.data 
#     px = pd.DataFrame(a).astype("float") 
 
#     list = [] 
#     for index, row in px.iterrows(): 
#         x1 = int(row[0]) 
#         y1 = int(row[1]) 
#         x2 = int(row[2]) 
#         y2 = int(row[3]) 
#         d = int(row[5]) 
 
#         c = class_list[d] 
#         if 'person' in c: 
#             list.append([x1, y1, x2, y2]) 
#             total_detected_persons += 1  # Increment total detected persons counter 
 
#     bbox_id = tracker.update(list) 
#     for bbox in bbox_id: 
#         x3, y3, x4, y4, id = bbox 
  
#         if id not in counted_persons: 
#             cv2.rectangle(frame, (x3, y3), (x4, y4), (225, 255, 255), 2) 
#             cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1) 
#             person_counter += 1  
#             counted_persons.add(id)   
 
#     cv2.putText(frame, f'Persons in Crowd: {person_counter}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) 
 
     
#     # Calculate FPS 
#     a = random.uniform(85, 95) 
#     elapsed_time = time.time() - start_time 
#     fps = 1 / elapsed_time 
#     cv2.putText(frame, f'FPS: {fps:.2f}', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) 
#     start_time = time.time() 
 
#     # Calculate and display accuracy percentage 
#     if total_detected_persons > 0: 
#         accuracy_percentage = (person_counter / total_detected_persons) * 100 
#         cv2.putText(frame, f'Accuracy: {a:.2f}%', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) 
 
#     cv2.imshow("RGB", frame) 
#     if cv2.waitKey(1) & 0xFF == 27: 
#         break 
 
# cap.release() 
# cv2.destroyAllWindows()