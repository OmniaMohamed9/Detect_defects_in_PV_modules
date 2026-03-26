from ultralytics import YOLO
import cv2

# === Paths ===
YOLO_MODEL_PATH = "yolov8n-cells/exp/weights/best.pt"
IMAGE_PATH = r'C:\Users\DELL\Downloads\crop.v1-roboflow-instant-1--eval-.yolov8\EL_final.png'
OUTPUT_PATH = "boxed_cells5.jpg"

# === Load YOLOv8 Model ===
yolo_model = YOLO(YOLO_MODEL_PATH)

# === Run YOLOv8 Inference ===
results = yolo_model(IMAGE_PATH)
img = cv2.imread(IMAGE_PATH)

for box in results[0].boxes.xyxy:
    x1, y1, x2, y2 = map(int, box)
    conf = results[0].boxes.conf[0] if hasattr(results[0].boxes, 'conf') else None
    label = "cell"
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

cv2.imwrite(OUTPUT_PATH, img)
print(f"Saved boxed image to {OUTPUT_PATH}") 