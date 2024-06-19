from ultralytics import YOLO
import cv2
import cvzone
import math

# video capture object
cap = cv2.VideoCapture(0)

# width
cap.set(3, 1280)

# height
cap.set(4, 720)

model = YOLO("../yolo-weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for result in results:
        boxes = result.boxes

        for box in boxes:
            # x1, y1, x2, y2 = box.xyxy[0]
            x1a, y1a, w, h = box.xywh[0]

            # Convert to coordinates
            # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            bbox = int(x1a), int(y1a), int(w), int(h)

            print(x1a, y1a, w, h)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cvzone.cornerRect(img, (x1a, y1a, w, h))

            confidence = math.ceil((box.conf[0] * 100)) / 100
            print(confidence)

            cvzone.putTextRect(
                img, f"{confidence}", (x1a, y1a - 20), (max(0, x1a), max(35, y1a))
            )

    cv2.imshow("Image", img)
    cv2.waitKey(1)
