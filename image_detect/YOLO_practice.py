from ultralytics import YOLO

model = YOLO("yolo26n.pt")
results = model("photo.jpg")
results[0].show()