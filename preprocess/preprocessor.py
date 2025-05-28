import sys
from edge_contours_detection import edge_detection, find_contours

def preprocess(img_path):
    image, edged = edge_detection(img_path)
    output = find_contours(image, edged)
    return output

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_preprocess.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    result = preprocess(image_path)


## Contouring the Cahracters
# contours= cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# contours = imutils.grab_contours(contours)
# contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
# screenCnt = None
    # for i, c in enumerate(contours):
    #     peri = cv2.arcLength(c, True)
    #     approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    #     print(f"Contour #{i}: {len(approx)} corners")
    #
    #     # Draw the contour for visualization
    #     temp = image.copy()
    #     cv2.drawContours(temp, [approx], -1, (0, 255, 0), 2)
    #     cv2.imshow(f"Contour #{i}", temp)
    #     cv2.waitKey(0)