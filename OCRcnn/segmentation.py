import cv2
import imutils
import sys
import numpy as np
from preprocessor import preprocess

def get_large_contours(input):
    contours= cv2.findContours(input.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

    return contours

def segmentate(input):
    contours = get_large_contours(input)
    filtered = []

    for i, c in enumerate(contours):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        if len(approx) > 4:
            filtered.append(approx) 
            print(f"Contour #{i}: {len(approx)} corners")
        
    filtered = sorted(filtered, key=lambda c: cv2.boundingRect(c)[0])    
        
    for i, f in enumerate(filtered):
        output = input.copy()
        cv2.drawContours(output, [f], -1, (0, 255, 0), 2)
        cv2.imshow(f"Filtered Contour #{i+1}", output)
        cv2.waitKey(0)
    
    return output, filtered

def extract_filtered_contours(image, filtered):
    padding = 5
    characters= []
    for contour in filtered:
        x, y, w, h = cv2.boundingRect(contour)
        roi = image[y-padding:y+h+padding, x-padding:x+w+padding]  # Crop character region
        roi = cv2.resize(roi, (32, 32))  # Resize to match your OCR input
        cv2.imshow(f"Extracted",roi)
        cv2.waitKey(0)
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=-1)# Add channel dim (H, W, 1)
        
        characters.append(roi)
    #return np.array(characters)
    return characters

def segmentater(img_path):
    input = preprocess(img_path)
    img, filtered = segmentate(input)
    result = extract_filtered_contours(img, filtered)

    return result

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_preprocess.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    input = preprocess(image_path)
    img, filtered = segmentate(input)
    extract_filtered_contours(img, filtered)
