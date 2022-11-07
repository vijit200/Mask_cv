import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os

#os.system('python detect.py --weights last.pt --img 416 --conf 0.5 --source 0')   #this is working fine

model = torch.hub.load(r'C:\Users\vijit\Mask_cv\yolov5-master','custom',source = 'local',path = r'C:\Users\vijit\Mask_cv\yolov5-master\last.pt',force_reload=True,device='cpu')

results = model(r'C:\Users\vijit\Mask_cv\istockphoto-1272058180-170667a.jpg')

results.print()

#having issue in this
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    # Make detections 
    

    frame = cv2.resize(frame,(416,416))

    results = model(frame)
    
    cv2.imshow('YOLO', np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()