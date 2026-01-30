from ultralytics import YOLO
model=YOLO("best.pt")
result=model("C:/Users/hw/Desktop/寒假项目/图片/",conf=0.1,save=True)
