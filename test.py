from ultralytics import YOLO

model = YOLO("./runs/detect/train4/weights/best.pt")

# predict results on test folder 
results = model.test(data="data.yaml", imgsz=640, batch=8, conf_thres=0.5, iou_thres=0.5)
print(results)
