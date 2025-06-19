from ultralytics import YOLO
import cv2
import os

# Set environment variable to avoid library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load a pretrained YOLOv8 model
model = YOLO("best.pt")

# Train the model
results = model.train(
    data=r"/Users/kapoor/Downloads/Student Monitoring System.v5i.yolov8/data.yaml",
    epochs=10,  # Reduce epochs to 1

)

# Validate the model
metrics = model.val()