
# 🧠 Attention Detection with YOLOv8

This project uses a custom-trained YOLOv8 model (`best.pt`) to detect human attention levels in images — specifically classifying between **Low Attention** and **High Attention**.

## 📸 Example Use Case
Analyzing frames from surveillance footage or classroom environments to detect students' attention levels in real-time.

---

## 🚀 Getting Started

### 🔧 Requirements
Install dependencies:
```bash
pip install ultralytics opencv-python matplotlib
````

### 📁 Folder Structure

```
.
├── best.pt                         # Custom trained YOLOv8 model
├── detect_attention.py             # Inference script
├── train/
│   └── images/
│       └── frame4739_jpg.rf.xxxx.jpg   # Input image sample
├── output.jpg                      # (Optional) Saved annotated output
└── README.md
```

---

## 🧠 Model Details

* Framework: [YOLOv8](https://docs.ultralytics.com/)
* Input: RGB image
* Output: Bounding boxes with:

  * **Class**: Low Attention / High Attention
  * **Confidence Score**

**Class Labels:**

```python
class_labels = ['Low Attention', 'High Attention']
```

---

## 🔍 Inference Script

```python
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load model
model = YOLO('best.pt')

# Image path
image_path = 'train/images/frame4739_jpg.rf.089ed45379350890c5e19adb74d6621e.jpg'

# Load image
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Predict
results = model.predict(image_path)
class_labels = ['Low Attention', 'High Attention']

# Annotate detections
for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    label = f"{class_labels[class_id]} ({confidence:.2f})"
    cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(image_rgb, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

# Show result
plt.imshow(image_rgb)
plt.axis('off')
plt.show()
```

---

## 💾 Save Output (Optional)

```python
cv2.imwrite("output.jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
```

---

## 🙋‍♂️ Author

**Bhumit Kumar Kapoor**
[GitHub: @kbhumik27](https://github.com/kbhumik27)

---

## 📌 Notes

* Ensure your model was trained on compatible image formats and annotations.
* You can expand this project to video frame inference or real-time webcam feeds.

---

## 🔗 References

* [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
* [OpenCV](https://opencv.org/)
* [Matplotlib](https://matplotlib.org/)

```

---

Let me know if you want a version that also includes how to train your model (`data.yaml`, `train.py`, etc.) or convert this to a Jupyter/Colab format!
```
