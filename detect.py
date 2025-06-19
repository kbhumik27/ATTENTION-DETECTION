import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load pre-trained YOLOv8 model
model = YOLO('best.pt')  # Make sure 'best.pt' is your trained model

# Load and preprocess the image
image_path = 'train/images/frame4739_jpg.rf.089ed45379350890c5e19adb74d6621e.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform inference
results = model.predict(image_path)

# Class labels (modify these based on your trained model)
class_labels = ['Low Attention', 'High Attention']

# Iterate through the results for each detection
for result in results[0].boxes:  # YOLO result stores detected boxes in a 'boxes' attribute
    
    # Extract bounding box coordinates
    box = result.xyxy[0].numpy()  # Get the bounding box [x1, y1, x2, y2]
    x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
    
    # Get the class ID and confidence score
    class_id = int(result.cls[0])  # Class ID (0 or 1, depending on your model)
    confidence = result.conf[0]  # Confidence score
    
    # Get the class label based on the class ID
    class_label = class_labels[class_id]
    
    # Draw bounding box on the image
    cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Put label with confidence score on the image
    label = f'{class_label} ({confidence:.2f})'
    cv2.putText(image_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the image with matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)
plt.axis('off')
plt.show()
