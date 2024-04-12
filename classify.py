from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-cls.pt')  # load an official model
# model = YOLO('path/to/best.pt')  # load a custom model

# Predict with the model
results = model("img/09.jpg", save=True)  # predict on an image