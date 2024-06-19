import cv2
from ultralytics import YOLO

#import v8 nano
model = YOLO('../yolo-weights/yolov8l.pt')

#Test run
results = model("images-test/image1.png", show=True)

cv2.waitKey(0)

results = model("images-test/image2.png", show=True)

cv2.waitKey(0)
