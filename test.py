from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO("yolov8n.pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# results = model.predict(source="0")
# results = model.predict(source="folder", show=True) # Display preds. Accepts all YOLO predict arguments

# from PIL
im1 = Image.open("img/10.jpg")
results = model.predict(source=im1, save=True)  # save plotted images
print(results[0].boxes)

# from ndarray
# im2 = cv2.imread("img/01.jpg")
# results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels

# from list of PIL/ndarray
# results = model.predict(source=[im1, im2])