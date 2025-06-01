from transform import four_point_transform
from skimage.filters import threshold_local
import argparse
import cv2
import imutils

def edge_detection(args):
	print("STEP 1: Edge Detection")
	# load the image and compute the ratio of the old height to the new height, clone it, and resize it
	image = cv2.imread(args)
	image = imutils.resize(image, height = 500)

	# convert the image to grayscale, blur it, and find edges in the image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 75, 200)

	# show the original image and the edge detected image
	#cv2.imshow("Image", image)
	#cv2.imshow("Edged", edged)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

	return image, edged

def find_plate(image, edged):
	print("STEP 2: Find contours of license plate")

	"""
	Key assumption: the license plate is the biggest rectangle object in the provide image.
	Find the top 5 biggest contour area, and loop through them until a contour with 4 corners is found.
	"""
	contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	contours = imutils.grab_contours(contours)
	contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
	screenCnt = None
	for contour in contours:
		peri = cv2.arcLength(contour, True)
		approx = cv2.approxPolyDP(contour, 0.02 * peri, True) # approximate the contour
		if len(approx) == 4:
			screenCnt = approx
			break

	# draw outline of the plate
	if(screenCnt is not None):
		cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
		#cv2.imshow("Outline", image)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()

		output = transform_perspective(image, screenCnt)
		return output

	else:
		"""
		In some images, the plate too big for the contour approximation, so step 3 must be skipped
		TODO: Try to find a fix
		"""
		T = threshold_local(edged, 11, offset=10, method="gaussian")
		output = (edged > T).astype("uint8") * 255
		#cv2.imshow("Original", imutils.resize(image.copy(), height=650))
		#cv2.imshow("Scanned", imutils.resize(output, height=650))
		#cv2.waitKey(0)
		return output

def transform_perspective(image, screenCnt):
	
	print("STEP 3: Apply perspective transform")
	ratio = image.shape[0] / 500.0 # parameter to scan text from original image
	#orig = image.copy()
	
	# apply the four point transform to obtain a straight view of the original image
	output = four_point_transform(image.copy(), screenCnt.reshape(4, 2) * ratio)

	"""		
	Convert the cropped image to grayscale, then threshold it to give it that 'black and white' effect
	Might need refactoring in the future.
	"""
	output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
	# T = threshold_local(output, 11, offset = 10, method = "gaussian")
	# output = (output > T).astype("uint8") * 255
	 # Use Otsu thresholding instead of adaptive
	_, output = cv2.threshold(output, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Optional: Fill gaps using morphology
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel)

	#cv2.imshow("Original", imutils.resize(image.copy(), height = 650))
	#cv2.imshow("Scanned", imutils.resize(output, height = 650))
	#cv2.waitKey(0)

	return output

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required = True,
		help = "Path to the image to be scanned")
	args = vars(ap.parse_args())
	image, edged = edge_detection(args)
	output = find_plate(image, edged)