import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from ultralytics import YOLO
model=YOLO("last.pt")
results=model.train(data='./data.yaml',epochs=100,imgsz=640,batch=8,device='cpu')
