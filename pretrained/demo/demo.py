from pathlib import Path
import numpy as np
import yolov5
import easyocr
import supervision as sv
import cv2
from google.colab.patches import cv2_imshow

# Load YOLOv5 model
model = yolov5.load('keremberke/yolov5n-license-plate')
model.conf = 0.25
model.iou = 0.45
model.agnostic = False
model.multi_label = False
model.max_det = 1000

# Load image
image_path = "/content/car1.jpg"
img = str(Path(image_path))
image = cv2.imread(image_path)

# Run YOLOv5 detection
results = model(img, size=640)
results = model(img, augment=True)  # TTA (optional)
# result.show()

# Initialize EasyOCR
reader = easyocr.Reader(['en'], model_storage_directory='/content', gpu=True)

# Prepare lists for visualization
xyxy, confidences, class_ids, labels = [], [], [], []
recognized_texts = []

# Process each detected bounding box
for pred in results.pred[0]:
    print("YOLO box:", pred[:4])
    x1, y1, x2, y2, conf, cls = map(int, pred[:6])

    # Crop detected license plate
    crop = image[y1:y2, x1:x2]
    cv2_imshow(crop)

    # OCR on cropped plate
    ocr_result = reader.readtext(crop)
    print("OCR result:", ocr_result)
    text = " ".join([entry[1] for entry in ocr_result]) if ocr_result else ""
    print(f"Cropped Box: ({x1}, {y1}) to ({x2}, {y2}) → OCR: '{text}'")

    # Save for display
    recognized_texts.append(text)
    xyxy.append([x1, y1, x2, y2])
    confidences.append(float(conf))
    class_ids.append(int(cls))
    labels.append(text)

# Visualization with Supervision
detections = sv.Detections(
    xyxy=np.array(xyxy),
    confidence=np.array(confidences),
    class_id=np.array(class_ids)
)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(text_position=sv.Position.BOTTOM_CENTER, text_scale=0.6)

annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

# Display and save result
sv.plot_image(annotated_image)
cv2.imwrite("Output.jpg", annotated_image)
cv2_imshow(cv2.imread("Output.jpg"))

# Print OCR results
print("Recognized License Plate Texts:")
for i, txt in enumerate(recognized_texts):
    print(f"Plate {i + 1}: {txt}")