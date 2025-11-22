from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *


cap = cv2.VideoCapture("OpenCv tests and projects/images and vids/4.mp4")
model = YOLO("yolov8x.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "f  risbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]
tracker = Sort(max_age = 20,min_hits = 3,iou_threshold=0.3)
limitsUp = [103,161,296,161]
limitsDown = [527,489,745,489]
totalCountsUp = []
totalCountsDown = []
mask = cv2.imread("OpenCv tests and projects/images and vids/mask_11.png", cv2.IMREAD_GRAYSCALE)
_, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
while True:
    success, img = cap.read()
    if not success:
        print("\nEND OF VIDEO\n")
        break
    imgGraphics = cv2.imread("OpenCv tests and projects/images and vids/graphics-1.png", cv2.IMREAD_UNCHANGED)
    scale_percent = 76
    width = int(imgGraphics.shape[1] * scale_percent / 100)
    height = int(imgGraphics.shape[0] * scale_percent / 100)
    imgGraphics = cv2.resize(imgGraphics, (width, height), interpolation=cv2.INTER_AREA)
    img = cvzone.overlayPNG(img, imgGraphics, (874, 260))
    imgRegion = cv2.bitwise_and(img, img, mask=mask)
    results = model(imgRegion,stream = True)
    detactions = np.empty((0, 5))
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if currentClass== "person" and conf >0.3:
                # cvzone.cornerRect(img, (x1, y1, w, h),l = 9,rt = 5,t = 5,colorR = (255, 200, 200),colorC=(0,0,128))
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), scale=0.70, thickness=1,colorT = (0,0,255),colorR = (255, 200, 200),colorB = (0, 0, 155),offset = 3)
              currentArray = np.array([x1, y1, x2, y2, conf])
              detactions = np.vstack((detactions,currentArray))
    results_tracker= tracker.update(detactions)
    cv2.line(img,(limitsUp[0],limitsUp[1]),(limitsUp[2],limitsUp[3]),(0,0,255),5)
    cv2.line(img,(limitsDown[0],limitsDown[1]),(limitsDown[2],limitsDown[3]),(0,0,255),5)
    for result in results_tracker:
        x1,y1,x2,y2,id = result
        print(result)
        x1, y1, x2, y2 = map(int,[x1,y1,x2,y2])
        w,h = x2-x1,y2-y1
        cvzone.cornerRect(img, (x1, y1, w, h),l = 9,rt = 5,t = 5,colorR = (255, 200, 200),colorC=(43,0,128))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3,colorT = (0,0,255),colorR = (255, 200, 200),colorB = (0, 0, 155),offset = 5)
        cx,cy = x1 + w//2, y1 +h//2
        cv2.circle(img,(cx,cy),5,(255,255,255),cv2.FILLED)
        if limitsUp[0]<cx< limitsUp[2] and limitsUp[1]- 15<cy<limitsUp[3]+ 15:
            if totalCountsUp.count(id) == 0:
                totalCountsUp.append(id)
                cv2.line(img,(limitsUp[0],limitsUp[1]),(limitsUp[2],limitsUp[3]),(0,255,0),5)
        # # cvzone.putTextRect(img, f'Count: {len(totalCounts)}', (50, 50)) 
    if limitsDown[0]<cx< limitsDown[2] and limitsDown[1]-15<cy<limitsDown[3]+ 15: #issue in here
        if totalCountsDown.count(id) == 0:#issue
                totalCountsDown.append(id)#issue
                cv2.line(img,(limitsDown[0],limitsDown[1]),(limitsDown[2],limitsDown[3]),(0,255,0),7)#issue
    
    cv2.putText(img,str(len(totalCountsUp)),(1025,335),cv2.FONT_HERSHEY_PLAIN,5,(150,200,50),7)
    cv2.putText(img,str(len(totalCountsDown)),(1217,335),cv2.FONT_HERSHEY_PLAIN,5,(0,0,255),5)#issue
    cv2.imshow("Car Detector", img)
    # cv2.imshow("image region", imgRegion)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        print("\nQUIT\n")
        break

    cv2.waitKey(1)
