from ultralytics import YOLO
import cv2
import cvzone

#video capture object
cap = cv2.VideoCapture(0)

#width
cap.set(3, 1280)

#height
cap.set(4, 720)

model = YOLO('../yolo-weights/yolov8n.pt')

while True:
    success, img = cap.read()
    cv2.imshow("Image", img)
    cv2.waitKey(1)