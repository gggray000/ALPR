import cv2
import numpy as np

# Load image
img = cv2.imread("../input/plate5.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Preprocess
blurred = cv2.bilateralFilter(gray, 11, 17, 17)  # reduce noise, keep edges
edged = cv2.Canny(blurred, 30, 200)

# Find contours
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Filter by shape
candidates = []
for cnt in contours:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

    if len(approx) >= 4:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = box.astype(np.intp)
        box = np.int32(box)

        w, h = rect[1]
        if w == 0 or h == 0:
            continue

        aspect_ratio = max(w, h) / min(w, h)
        area = cv2.contourArea(cnt)

        if 2 < aspect_ratio < 6 and area > 1000:
            candidates.append(box)

# Draw results
output = img.copy()
for box in candidates:
    cv2.drawContours(output, [box], 0, (0, 255, 0), 2)

cv2.imshow("Plate Candidates", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
