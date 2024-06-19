import cv2
from ultralytics import YOLO

#import v8 nano
model = YOLO('yolov8n.pt')

#Test run
results = model("images-test/image1.png", show=True)

cv2.waitKey()

results = model("images-test/image2.png", show=True)

cv2.waitKey()
