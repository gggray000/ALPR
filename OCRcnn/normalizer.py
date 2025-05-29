import sys
from edge_contours_detection import edge_detection, find_plate

def normalize(img_path):
    image, edged = edge_detection(img_path)
    output = find_plate(image, edged)
    return output

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_preprocess.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    result = normalize(image_path)
